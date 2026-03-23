/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const log = require('../../log');

class ShaderProgram {
	/**
	 * @param {GLContext|GPUContext} ctx
	 * @param {string} vert_source - GLSL vertex shader or WGSL module source
	 * @param {string} [frag_source] - GLSL fragment shader (omit for WebGPU, where vert_source is the full WGSL module)
	 */
	constructor(ctx, vert_source, frag_source) {
		this.ctx = ctx;
		this.gl = ctx.gl;
		this.program = null;
		this.uniform_locations = new Map();
		this.uniform_block_indices = new Map();

		if (ctx.is_webgpu) {
			this._uniforms = new Map();
			this._uniform_buffer = null;
			this._uniform_bind_group = null;
			this._uniform_dirty = true;
			this._pipeline = null;
			this._bind_group_layout = null;
			this._pipeline_layout = null;
			this._next_uniform_offset = 0;
			this._cpu_buffer = null;
			this._texture_bind_group_layout = null;
			this._compile_wgsl(vert_source);
		} else {
			this._compile(vert_source, frag_source);
		}
	}

	/**
	 * @param {string} vert_source
	 * @param {string} frag_source
	 */
	_compile(vert_source, frag_source) {
		const gl = this.gl;

		const vert_shader = this._compile_shader(gl.VERTEX_SHADER, vert_source);
		const frag_shader = this._compile_shader(gl.FRAGMENT_SHADER, frag_source);

		if (!vert_shader || !frag_shader)
			return;

		const program = gl.createProgram();
		gl.attachShader(program, vert_shader);
		gl.attachShader(program, frag_shader);
		gl.linkProgram(program);

		// shaders can be deleted after linking
		gl.deleteShader(vert_shader);
		gl.deleteShader(frag_shader);

		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			const info = gl.getProgramInfoLog(program);
			log.write('Shader program link error: %s', info);
			gl.deleteProgram(program);
			return;
		}

		this.program = program;
	}

	/**
	 * @param {number} type
	 * @param {string} source
	 * @returns {WebGLShader|null}
	 */
	_compile_shader(type, source) {
		const gl = this.gl;
		const shader = gl.createShader(type);

		gl.shaderSource(shader, source);
		gl.compileShader(shader);

		if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
			const info = gl.getShaderInfoLog(shader);
			const type_name = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
			log.write('Shader compile error (%s): %s', type_name, info);
			gl.deleteShader(shader);
			return null;
		}

		return shader;
	}

	// ── WebGPU backend ──────────────────────────────────────────────

	/**
	 * Compile a WGSL shader module.
	 * @param {string} wgsl_source
	 */
	_compile_wgsl(wgsl_source) {
		const device = this.ctx.device;

		this._shader_module = device.createShaderModule({ code: wgsl_source });
		this._shader_module.getCompilationInfo().then(info => {
			for (const msg of info.messages) {
				if (msg.type === 'error')
					log.write('WGSL compile error (line %d): %s', msg.lineNum, msg.message);
				else if (msg.type === 'warning')
					log.write('WGSL compile warning (line %d): %s', msg.lineNum, msg.message);
			}
		});

		this._wgsl_source = wgsl_source;
		this.program = this._shader_module;
	}

	/**
	 * Creates a render pipeline for this shader. Call once after setting up vertex layout.
	 * @param {object} descriptor - Pipeline descriptor overrides
	 * @param {GPUVertexBufferLayout[]} descriptor.vertex_buffers - Vertex buffer layouts
	 * @param {string} [descriptor.vertex_entry='vs_main'] - Vertex shader entry point
	 * @param {string} [descriptor.fragment_entry='fs_main'] - Fragment shader entry point
	 * @param {string} [descriptor.topology='triangle-list'] - Primitive topology
	 * @param {object} [descriptor.depth_stencil] - Depth/stencil state (defaults from context)
	 * @param {object} [descriptor.blend] - Blend state
	 * @param {string} [descriptor.cull_mode] - Cull mode (defaults from context)
	 * @param {GPUBindGroupLayout[]} [descriptor.extra_bind_group_layouts] - Additional bind group layouts
	 * @returns {GPURenderPipeline}
	 */
	create_pipeline(descriptor = {}) {
		const device = this.ctx.device;

		// bind group 0: uniform buffer
		this._bind_group_layout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: { type: 'uniform' }
			}]
		});

		const bind_group_layouts = [this._bind_group_layout];
		if (this._texture_bind_group_layout)
			bind_group_layouts.push(this._texture_bind_group_layout);
		if (descriptor.extra_bind_group_layouts)
			bind_group_layouts.push(...descriptor.extra_bind_group_layouts);

		this._pipeline_layout = device.createPipelineLayout({
			bindGroupLayouts: bind_group_layouts
		});

		const fragment_targets = [{
			format: this.ctx.format,
			blend: descriptor.blend ?? this.ctx.get_blend_state()
		}];

		// remove undefined blend (no-blend = no blend field)
		if (fragment_targets[0].blend === undefined)
			delete fragment_targets[0].blend;

		this._pipeline = device.createRenderPipeline({
			layout: this._pipeline_layout,
			vertex: {
				module: this._shader_module,
				entryPoint: descriptor.vertex_entry ?? 'vs_main',
				buffers: descriptor.vertex_buffers ?? []
			},
			fragment: {
				module: this._shader_module,
				entryPoint: descriptor.fragment_entry ?? 'fs_main',
				targets: fragment_targets
			},
			primitive: {
				topology: descriptor.topology ?? 'triangle-list',
				cullMode: descriptor.cull_mode ?? this.ctx.get_cull_mode()
			},
			depthStencil: descriptor.depth_stencil ?? this.ctx.get_depth_stencil_state(),
			multisample: {
				count: this.ctx.sample_count
			}
		});

		return this._pipeline;
	}

	/**
	 * Get or create the pipeline. Returns null if not yet created.
	 * @returns {GPURenderPipeline|null}
	 */
	get pipeline() {
		return this._pipeline;
	}

	/**
	 * Allocate (or resize) the GPU uniform buffer to hold at least `size` bytes.
	 * @param {number} size - Minimum buffer size in bytes (will be rounded up to 16-byte alignment)
	 */
	_ensure_uniform_buffer(size) {
		const aligned = Math.ceil(size / 16) * 16;

		if (this._uniform_buffer && this._uniform_buffer.size >= aligned)
			return;

		if (this._uniform_buffer)
			this._uniform_buffer.destroy();

		this._uniform_buffer = this.ctx.device.createBuffer({
			size: aligned,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// bind group must be recreated when buffer changes
		this._uniform_bind_group = null;
	}

	/**
	 * Flush all pending uniform values to the GPU buffer and update the bind group.
	 * Call this before issuing draw commands.
	 */
	flush_uniforms() {
		if (!this.ctx.is_webgpu || this._uniforms.size === 0)
			return;

		// compute total buffer size from uniform entries
		let total_size = 0;
		for (const entry of this._uniforms.values())
			total_size = Math.max(total_size, entry.offset + entry.size);

		this._ensure_uniform_buffer(total_size);

		// build a single CPU buffer and write all uniforms into it
		const aligned_size = Math.ceil(total_size / 16) * 16;
		if (!this._cpu_buffer || this._cpu_buffer.byteLength < aligned_size)
			this._cpu_buffer = new Uint8Array(aligned_size);

		for (const entry of this._uniforms.values())
			this._cpu_buffer.set(entry.data, entry.offset);

		this.ctx.device.queue.writeBuffer(this._uniform_buffer, 0, this._cpu_buffer.buffer, 0, aligned_size);

		// auto-create bind group layout if not yet created
		if (!this._bind_group_layout) {
			this._bind_group_layout = this.ctx.device.createBindGroupLayout({
				entries: [{
					binding: 0,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: 'uniform' }
				}]
			});
		}

		// recreate bind group if needed
		if (!this._uniform_bind_group) {
			this._uniform_bind_group = this.ctx.device.createBindGroup({
				layout: this._bind_group_layout,
				entries: [{
					binding: 0,
					resource: { buffer: this._uniform_buffer }
				}]
			});
		}

		this._uniform_dirty = false;
	}

	/**
	 * Returns the bind group for uniform buffer (group 0).
	 * @returns {GPUBindGroup|null}
	 */
	get uniform_bind_group() {
		return this._uniform_bind_group;
	}

	/**
	 * Create a bind group layout for textures (group 1).
	 * @param {number} texture_count - Number of texture+sampler pairs
	 * @returns {GPUBindGroupLayout}
	 */
	create_texture_bind_group_layout(texture_count) {
		const entries = [];
		for (let i = 0; i < texture_count; i++) {
			entries.push({
				binding: i * 2,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: 'float' }
			});
			entries.push({
				binding: i * 2 + 1,
				visibility: GPUShaderStage.FRAGMENT,
				sampler: { type: 'filtering' }
			});
		}

		this._texture_bind_group_layout = this.ctx.device.createBindGroupLayout({ entries });
		return this._texture_bind_group_layout;
	}

	/**
	 * Create a texture bind group from GLTexture objects.
	 * @param {GLTexture[]} textures - Array of GLTexture objects (must have .texture and .sampler)
	 * @returns {GPUBindGroup}
	 */
	create_texture_bind_group(textures) {
		if (!this._texture_bind_group_layout)
			throw new Error('Texture bind group layout not created');

		const entries = [];
		for (let i = 0; i < textures.length; i++) {
			const tex = textures[i];
			entries.push({
				binding: i * 2,
				resource: tex.texture.createView()
			});
			entries.push({
				binding: i * 2 + 1,
				resource: tex.sampler
			});
		}

		return this.ctx.device.createBindGroup({
			layout: this._texture_bind_group_layout,
			entries
		});
	}

	/**
	 * Register a uniform with a specific byte offset in the buffer.
	 * This tells the shader program where each named uniform lives in the uniform buffer.
	 * @param {string} name
	 * @param {number} offset - Byte offset in the uniform buffer
	 * @param {number} size - Size in bytes
	 */
	define_uniform(name, offset, size) {
		const existing = this._uniforms.get(name);
		if (existing && existing.offset === offset && existing.size === size)
			return;

		this._uniforms.set(name, {
			offset,
			size,
			data: new Uint8Array(size)
		});
		this._uniform_dirty = true;
	}

	/**
	 * Auto-define a uniform on first use with WGSL-compatible alignment.
	 * @param {string} name
	 * @param {number} byte_size - Size in bytes of the data
	 */
	_auto_define_uniform(name, byte_size) {
		if (this._uniforms.has(name))
			return;

		// WGSL alignment: vec4/mat4 align to 16, vec2 to 8, scalars to 4
		const alignment = byte_size >= 16 ? 16 : (byte_size >= 8 ? 8 : 4);
		let offset = Math.ceil(this._next_uniform_offset / alignment) * alignment;

		this.define_uniform(name, offset, byte_size);
		this._next_uniform_offset = offset + byte_size;
	}

	/**
	 * Store a uniform value for later upload. Auto-defines the uniform on first use.
	 * Uses raw byte copy to preserve integer bit patterns correctly.
	 * @param {string} name
	 * @param {Float32Array|Int32Array|number[]} data
	 */
	_set_gpu_uniform(name, data) {
		const byte_size = (data.byteLength !== undefined) ? data.byteLength : data.length * 4;
		this._auto_define_uniform(name, byte_size);

		const entry = this._uniforms.get(name);
		if (!entry)
			return;

		if (data.buffer) {
			// Typed array - copy raw bytes to preserve int/float bit patterns
			const src = new Uint8Array(data.buffer, data.byteOffset, Math.min(data.byteLength, entry.size));
			entry.data.set(src, 0);
		} else if (Array.isArray(data)) {
			// Plain array - treat as float32
			const tmp = new Float32Array(data);
			const src = new Uint8Array(tmp.buffer, 0, Math.min(tmp.byteLength, entry.size));
			entry.data.set(src, 0);
		}

		this._uniform_dirty = true;
	}

	// ── Common API (works for both backends) ────────────────────────

	/**
	 * @returns {boolean}
	 */
	is_valid() {
		return this.program !== null;
	}

	use() {
		if (this.ctx.is_webgpu) {
			// for WebGPU, pipeline binding happens on the render pass encoder
			// this is kept for API compatibility; renderers should call
			// pass.setPipeline(shader.pipeline) directly
			this.ctx.use_program(this._pipeline);
		} else {
			this.ctx.use_program(this.program);
		}
	}

	/**
	 * @param {string} name
	 * @returns {WebGLUniformLocation|number|null}
	 */
	get_uniform_location(name) {
		if (this.ctx.is_webgpu) {
			// return the byte offset if the uniform is defined, otherwise null
			const entry = this._uniforms.get(name);
			return entry ? entry.offset : null;
		}

		if (this.uniform_locations.has(name))
			return this.uniform_locations.get(name);

		const location = this.gl.getUniformLocation(this.program, name);
		this.uniform_locations.set(name, location);
		return location;
	}

	/**
	 * @param {string} name
	 * @returns {number}
	 */
	get_uniform_block_index(name) {
		if (this.ctx.is_webgpu)
			return 0; // bind groups handle this in WebGPU

		if (this.uniform_block_indices.has(name))
			return this.uniform_block_indices.get(name);

		const index = this.gl.getUniformBlockIndex(this.program, name);
		this.uniform_block_indices.set(name, index);
		return index;
	}

	/**
	 * @param {string} name
	 * @param {number} binding_point
	 */
	bind_uniform_block(name, binding_point) {
		if (this.ctx.is_webgpu)
			return; // bind groups handle this in WebGPU

		const index = this.get_uniform_block_index(name);
		if (index !== this.gl.INVALID_INDEX)
			this.gl.uniformBlockBinding(this.program, index, binding_point);
	}

	/**
	 * @param {string} name
	 * @param {number} value
	 */
	set_uniform_1i(name, value) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_uniform(name, new Int32Array([value]));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniform1i(loc, value);
	}

	/**
	 * @param {string} name
	 * @param {number} value
	 */
	set_uniform_1f(name, value) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_uniform(name, new Float32Array([value]));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniform1f(loc, value);
	}

	/**
	 * @param {string} name
	 * @param {number} x
	 * @param {number} y
	 */
	set_uniform_2f(name, x, y) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_uniform(name, new Float32Array([x, y]));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniform2f(loc, x, y);
	}

	/**
	 * @param {string} name
	 * @param {number} x
	 * @param {number} y
	 * @param {number} z
	 */
	set_uniform_3f(name, x, y, z) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_uniform(name, new Float32Array([x, y, z]));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniform3f(loc, x, y, z);
	}

	/**
	 * @param {string} name
	 * @param {number} x
	 * @param {number} y
	 * @param {number} z
	 * @param {number} w
	 */
	set_uniform_4f(name, x, y, z, w) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_uniform(name, new Float32Array([x, y, z, w]));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniform4f(loc, x, y, z, w);
	}

	/**
	 * @param {string} name
	 * @param {Float32Array|number[]} value
	 */
	set_uniform_3fv(name, value) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_uniform(name, value instanceof Float32Array ? value : new Float32Array(value));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniform3fv(loc, value);
	}

	/**
	 * @param {string} name
	 * @param {Float32Array|number[]} value
	 */
	set_uniform_4fv(name, value) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_uniform(name, value instanceof Float32Array ? value : new Float32Array(value));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniform4fv(loc, value);
	}

	/**
	 * @param {string} name
	 * @param {boolean} transpose
	 * @param {Float32Array|number[]} value
	 */
	set_uniform_mat3(name, transpose, value) {
		if (this.ctx.is_webgpu) {
			// WebGPU does not support transpose; caller must pre-transpose if needed
			this._set_gpu_uniform(name, value instanceof Float32Array ? value : new Float32Array(value));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniformMatrix3fv(loc, transpose, value);
	}

	/**
	 * @param {string} name
	 * @param {boolean} transpose
	 * @param {Float32Array|number[]} value
	 */
	set_uniform_mat4(name, transpose, value) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_uniform(name, value instanceof Float32Array ? value : new Float32Array(value));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniformMatrix4fv(loc, transpose, value);
	}

	/**
	 * @param {string} name
	 * @param {boolean} transpose
	 * @param {Float32Array|number[]} value
	 */
	set_uniform_mat4_array(name, transpose, value) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_uniform(name, value instanceof Float32Array ? value : new Float32Array(value));
			return;
		}

		const loc = this.get_uniform_location(name);
		if (loc !== null)
			this.gl.uniformMatrix4fv(loc, transpose, value);
	}

	/**
	 * Recompile shader with new source (hot-reload)
	 * @param {string} vert_source - GLSL vertex shader or WGSL module source
	 * @param {string} [frag_source] - GLSL fragment shader (unused for WebGPU)
	 * @returns {boolean}
	 */
	recompile(vert_source, frag_source) {
		if (this.ctx.is_webgpu) {
			this._compile_wgsl(vert_source);
			// pipeline must be recreated by the renderer after recompile
			this._pipeline = null;
			this._uniform_bind_group = null;
			return this.program !== null;
		}

		const gl = this.gl;

		const vert_shader = this._compile_shader(gl.VERTEX_SHADER, vert_source);
		const frag_shader = this._compile_shader(gl.FRAGMENT_SHADER, frag_source);

		if (!vert_shader || !frag_shader) {
			if (vert_shader)
				gl.deleteShader(vert_shader);

			if (frag_shader)
				gl.deleteShader(frag_shader);

			return false;
		}

		const new_program = gl.createProgram();
		gl.attachShader(new_program, vert_shader);
		gl.attachShader(new_program, frag_shader);
		gl.linkProgram(new_program);

		gl.deleteShader(vert_shader);
		gl.deleteShader(frag_shader);

		if (!gl.getProgramParameter(new_program, gl.LINK_STATUS)) {
			const info = gl.getProgramInfoLog(new_program);
			const log = require('../../log');
			log.write('Shader program link error on recompile: %s', info);
			gl.deleteProgram(new_program);
			return false;
		}

		// delete old program and swap in new one
		if (this.program)
			gl.deleteProgram(this.program);

		this.program = new_program;

		// clear uniform caches since locations change
		this.uniform_locations.clear();
		this.uniform_block_indices.clear();

		return true;
	}

	dispose() {
		// unregister from Shaders module if tracked
		if (this._shader_name) {
			const Shaders = require('../Shaders');
			Shaders.unregister(this);
		}

		if (this.ctx.is_webgpu) {
			if (this._uniform_buffer) {
				this._uniform_buffer.destroy();
				this._uniform_buffer = null;
			}

			this._uniform_bind_group = null;
			this._pipeline = null;
			this._bind_group_layout = null;
			this._pipeline_layout = null;
			this._texture_bind_group_layout = null;
			this._shader_module = null;
			this._cpu_buffer = null;
			this.program = null;
			this._uniforms.clear();
		} else {
			if (this.program) {
				this.gl.deleteProgram(this.program);
				this.program = null;
			}

			this.uniform_locations.clear();
			this.uniform_block_indices.clear();
		}
	}
}

module.exports = ShaderProgram;
