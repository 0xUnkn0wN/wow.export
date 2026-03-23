/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

// attribute location constants matching shader layout
const AttributeLocation = {
	POSITION: 0,
	NORMAL: 1,
	BONE_INDICES: 2,
	BONE_WEIGHTS: 3,
	TEXCOORD: 4,
	TEXCOORD2: 5,
	COLOR: 6,
	COLOR2: 7,
	TEXCOORD3: 8,
	TEXCOORD4: 9,
	COLOR3: 10
};

class VertexArray {
	/**
	 * @param {GLContext|GPUContext} ctx
	 */
	constructor(ctx) {
		this.ctx = ctx;
		this.gl = ctx.gl;
		this.index_count = 0;

		if (ctx.is_webgpu) {
			this.vao = null;
			this.vbo = null;
			this.ebo = null;
			this._gpu_vbos = [];
			this._gpu_ebo = null;
			this._vertex_layouts = [];
			this.index_type = 'uint16';
			this._index_bytes = 2;
		} else {
			this.vao = this.gl.createVertexArray();
			this.vbo = null;
			this.ebo = null;
			this.index_type = this.gl.UNSIGNED_SHORT;
		}
	}

	bind() {
		if (this.ctx.is_webgpu)
			return; // WebGPU binds buffers on the render pass

		this.ctx.bind_vao(this.vao);
	}

	/**
	 * Create and upload vertex buffer
	 * @param {Float32Array|ArrayBuffer} data
	 * @param {number} [usage=gl.STATIC_DRAW]
	 */
	set_vertex_buffer(data, usage) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_vertex_buffer(data);
			return;
		}

		const gl = this.gl;
		usage = usage ?? gl.STATIC_DRAW;

		if (!this.vbo)
			this.vbo = gl.createBuffer();

		gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
		gl.bufferData(gl.ARRAY_BUFFER, data, usage);
	}

	/**
	 * Create and upload index buffer
	 * @param {Uint16Array|Uint32Array} data
	 * @param {number} [usage=gl.STATIC_DRAW]
	 */
	set_index_buffer(data, usage) {
		if (this.ctx.is_webgpu) {
			this._set_gpu_index_buffer(data);
			return;
		}

		const gl = this.gl;
		usage = usage ?? gl.STATIC_DRAW;

		if (!this.ebo)
			this.ebo = gl.createBuffer();

		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, usage);

		this.index_count = data.length;

		if (data instanceof Uint32Array)
			this.index_type = gl.UNSIGNED_INT;
		else
			this.index_type = gl.UNSIGNED_SHORT;
	}

	/**
	 * Set up vertex attribute
	 * @param {number} location - attribute location
	 * @param {number} size - number of components (1-4)
	 * @param {number} type - gl.FLOAT, gl.UNSIGNED_BYTE, etc
	 * @param {boolean} normalized
	 * @param {number} stride - byte stride
	 * @param {number} offset - byte offset
	 */
	set_attribute(location, size, type, normalized, stride, offset) {
		if (this.ctx.is_webgpu)
			return; // handled by vertex layout descriptors

		const gl = this.gl;

		gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
		gl.enableVertexAttribArray(location);
		gl.vertexAttribPointer(location, size, type, normalized, stride, offset);
	}

	/**
	 * Set up integer vertex attribute (for bone indices)
	 * @param {number} location
	 * @param {number} size
	 * @param {number} type
	 * @param {number} stride
	 * @param {number} offset
	 */
	set_attribute_i(location, size, type, stride, offset) {
		if (this.ctx.is_webgpu)
			return; // handled by vertex layout descriptors

		const gl = this.gl;

		gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
		gl.enableVertexAttribArray(location);
		gl.vertexAttribIPointer(location, size, type, stride, offset);
	}

	/**
	 * Set up M2 vertex format
	 * layout: position(3f) + normal(3f) + bone_indices(4ub) + bone_weights(4ub) + uv1(2f)
	 * stride = 40 bytes
	 */
	setup_m2_vertex_format() {
		if (this.ctx.is_webgpu) {
			this._vertex_layouts = [this._get_m2_vertex_layout()];
			return;
		}

		const gl = this.gl;
		const stride = 40;

		this.bind();
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);

		// position: vec3 at offset 0
		gl.enableVertexAttribArray(AttributeLocation.POSITION);
		gl.vertexAttribPointer(AttributeLocation.POSITION, 3, gl.FLOAT, false, stride, 0);

		// normal: vec3 at offset 12
		gl.enableVertexAttribArray(AttributeLocation.NORMAL);
		gl.vertexAttribPointer(AttributeLocation.NORMAL, 3, gl.FLOAT, false, stride, 12);

		// bone indices: uvec4 at offset 24
		gl.enableVertexAttribArray(AttributeLocation.BONE_INDICES);
		gl.vertexAttribIPointer(AttributeLocation.BONE_INDICES, 4, gl.UNSIGNED_BYTE, stride, 24);

		// bone weights: vec4 (normalized) at offset 28
		gl.enableVertexAttribArray(AttributeLocation.BONE_WEIGHTS);
		gl.vertexAttribPointer(AttributeLocation.BONE_WEIGHTS, 4, gl.UNSIGNED_BYTE, true, stride, 28);

		// texcoord1: vec2 at offset 32
		gl.enableVertexAttribArray(AttributeLocation.TEXCOORD);
		gl.vertexAttribPointer(AttributeLocation.TEXCOORD, 2, gl.FLOAT, false, stride, 32);

		// texcoord2 uses separate buffer or not present for simple models
	}

	/**
	 * Set up M2 vertex format with separate buffers (matching current loader structure)
	 * @param {WebGLBuffer|GPUBuffer} pos_buffer
	 * @param {WebGLBuffer|GPUBuffer} norm_buffer
	 * @param {WebGLBuffer|GPUBuffer} uv_buffer
	 * @param {WebGLBuffer|GPUBuffer} bone_idx_buffer
	 * @param {WebGLBuffer|GPUBuffer} bone_weight_buffer
	 * @param {WebGLBuffer|GPUBuffer} [uv2_buffer]
	 */
	setup_m2_separate_buffers(pos_buffer, norm_buffer, uv_buffer, bone_idx_buffer, bone_weight_buffer, uv2_buffer) {
		if (this.ctx.is_webgpu) {
			this._gpu_vbos = [pos_buffer, norm_buffer, bone_idx_buffer, bone_weight_buffer, uv_buffer];
			if (uv2_buffer)
				this._gpu_vbos.push(uv2_buffer);

			this._vertex_layouts = this._get_m2_separate_vertex_layouts(!!uv2_buffer);
			return;
		}

		const gl = this.gl;

		this.bind();

		// position: vec3
		gl.bindBuffer(gl.ARRAY_BUFFER, pos_buffer);
		gl.enableVertexAttribArray(AttributeLocation.POSITION);
		gl.vertexAttribPointer(AttributeLocation.POSITION, 3, gl.FLOAT, false, 0, 0);

		// normal: vec3
		gl.bindBuffer(gl.ARRAY_BUFFER, norm_buffer);
		gl.enableVertexAttribArray(AttributeLocation.NORMAL);
		gl.vertexAttribPointer(AttributeLocation.NORMAL, 3, gl.FLOAT, false, 0, 0);

		// bone indices: uvec4
		gl.bindBuffer(gl.ARRAY_BUFFER, bone_idx_buffer);
		gl.enableVertexAttribArray(AttributeLocation.BONE_INDICES);
		gl.vertexAttribIPointer(AttributeLocation.BONE_INDICES, 4, gl.UNSIGNED_BYTE, 0, 0);

		// bone weights: vec4 normalized
		gl.bindBuffer(gl.ARRAY_BUFFER, bone_weight_buffer);
		gl.enableVertexAttribArray(AttributeLocation.BONE_WEIGHTS);
		gl.vertexAttribPointer(AttributeLocation.BONE_WEIGHTS, 4, gl.UNSIGNED_BYTE, true, 0, 0);

		// texcoord1: vec2
		gl.bindBuffer(gl.ARRAY_BUFFER, uv_buffer);
		gl.enableVertexAttribArray(AttributeLocation.TEXCOORD);
		gl.vertexAttribPointer(AttributeLocation.TEXCOORD, 2, gl.FLOAT, false, 0, 0);

		// texcoord2: vec2 (optional)
		if (uv2_buffer) {
			gl.bindBuffer(gl.ARRAY_BUFFER, uv2_buffer);
			gl.enableVertexAttribArray(AttributeLocation.TEXCOORD2);
			gl.vertexAttribPointer(AttributeLocation.TEXCOORD2, 2, gl.FLOAT, false, 0, 0);
		} else {
			gl.disableVertexAttribArray(AttributeLocation.TEXCOORD2);
		}
	}

	/**
	 * Set up WMO vertex format
	 * layout: position(3f) + normal(3f) + uv1(2f) + color(4ub) + color2(4ub) + color3(4ub) + uv2(2f) + uv3(2f) + uv4(2f)
	 */
	setup_wmo_separate_buffers(pos_buffer, norm_buffer, uv_buffer, color_buffer, color2_buffer, color3_buffer, uv2_buffer, uv3_buffer, uv4_buffer) {
		if (this.ctx.is_webgpu) {
			this._gpu_vbos = [pos_buffer, norm_buffer, uv_buffer];
			const optional = [color_buffer, color2_buffer, color3_buffer, uv2_buffer, uv3_buffer, uv4_buffer];
			for (const buf of optional) {
				if (buf) this._gpu_vbos.push(buf);
			}
			this._vertex_layouts = this._get_wmo_separate_vertex_layouts(!!color_buffer, !!color2_buffer, !!color3_buffer, !!uv2_buffer, !!uv3_buffer, !!uv4_buffer);
			return;
		}

		const gl = this.gl;

		this.bind();

		// position: vec3
		gl.bindBuffer(gl.ARRAY_BUFFER, pos_buffer);
		gl.enableVertexAttribArray(AttributeLocation.POSITION);
		gl.vertexAttribPointer(AttributeLocation.POSITION, 3, gl.FLOAT, false, 0, 0);

		// normal: vec3
		gl.bindBuffer(gl.ARRAY_BUFFER, norm_buffer);
		gl.enableVertexAttribArray(AttributeLocation.NORMAL);
		gl.vertexAttribPointer(AttributeLocation.NORMAL, 3, gl.FLOAT, false, 0, 0);

		// texcoord1: vec2
		gl.bindBuffer(gl.ARRAY_BUFFER, uv_buffer);
		gl.enableVertexAttribArray(AttributeLocation.TEXCOORD);
		gl.vertexAttribPointer(AttributeLocation.TEXCOORD, 2, gl.FLOAT, false, 0, 0);

		// disable unused
		gl.disableVertexAttribArray(AttributeLocation.BONE_INDICES);
		gl.disableVertexAttribArray(AttributeLocation.BONE_WEIGHTS);

		// vertex color (optional)
		if (color_buffer) {
			gl.bindBuffer(gl.ARRAY_BUFFER, color_buffer);
			gl.enableVertexAttribArray(AttributeLocation.COLOR);
			gl.vertexAttribPointer(AttributeLocation.COLOR, 4, gl.UNSIGNED_BYTE, true, 0, 0);
		} else {
			gl.disableVertexAttribArray(AttributeLocation.COLOR);
		}

		if(color2_buffer){
			gl.bindBuffer(gl.ARRAY_BUFFER, color2_buffer);
			gl.enableVertexAttribArray(AttributeLocation.COLOR2);
			gl.vertexAttribPointer(AttributeLocation.COLOR2, 4, gl.UNSIGNED_BYTE, true, 0, 0);
		} else {
			gl.disableVertexAttribArray(AttributeLocation.COLOR2);
		}

		// color 3 (optional)
		if (color3_buffer) {
			gl.bindBuffer(gl.ARRAY_BUFFER, color3_buffer);
			gl.enableVertexAttribArray(AttributeLocation.COLOR3);
			gl.vertexAttribPointer(AttributeLocation.COLOR3, 4, gl.UNSIGNED_BYTE, true, 0, 0);
		} else {
			gl.disableVertexAttribArray(AttributeLocation.COLOR3);
		}

		// uv2 (optional)
		if (uv2_buffer) {
			gl.bindBuffer(gl.ARRAY_BUFFER, uv2_buffer);
			gl.enableVertexAttribArray(AttributeLocation.TEXCOORD2);
			gl.vertexAttribPointer(AttributeLocation.TEXCOORD2, 2, gl.FLOAT, false, 0, 0);
		} else {
			gl.disableVertexAttribArray(AttributeLocation.TEXCOORD2);
		}

		// uv3 (optional)
		if (uv3_buffer) {
			gl.bindBuffer(gl.ARRAY_BUFFER, uv3_buffer);
			gl.enableVertexAttribArray(AttributeLocation.TEXCOORD3);
			gl.vertexAttribPointer(AttributeLocation.TEXCOORD3, 2, gl.FLOAT, false, 0, 0);
		} else {
			gl.disableVertexAttribArray(AttributeLocation.TEXCOORD3);
		}

		// uv4 (optional)
		if (uv4_buffer) {
			gl.bindBuffer(gl.ARRAY_BUFFER, uv4_buffer);
			gl.enableVertexAttribArray(AttributeLocation.TEXCOORD4);
			gl.vertexAttribPointer(AttributeLocation.TEXCOORD4, 2, gl.FLOAT, false, 0, 0);
		} else {
			gl.disableVertexAttribArray(AttributeLocation.TEXCOORD4);
		}
	}

	/**
	 * @param {number} mode
	 * @param {number} [count]
	 * @param {number} [offset=0]
	 */
	draw(mode, count, offset = 0) {
		if (this.ctx.is_webgpu) {
			this._draw_gpu(count, offset);
			return;
		}

		count = count ?? this.index_count;
		this.ctx.draw_elements(mode, count, this.index_type, offset * (this.index_type === this.gl.UNSIGNED_INT ? 4 : 2));
	}

	dispose() {
		if (this.ctx.is_webgpu) {
			for (const buf of this._gpu_vbos) {
				if (buf && buf.destroy)
					buf.destroy();
			}
			this._gpu_vbos = [];

			if (this._gpu_ebo) {
				this._gpu_ebo.destroy();
				this._gpu_ebo = null;
			}
			return;
		}

		const gl = this.gl;

		if (this.vao) {
			gl.deleteVertexArray(this.vao);
			this.vao = null;
		}

		if (this.vbo) {
			gl.deleteBuffer(this.vbo);
			this.vbo = null;
		}

		if (this.ebo) {
			gl.deleteBuffer(this.ebo);
			this.ebo = null;
		}
	}

	// ── WebGPU helpers ──────────────────────────────────────────────

	/**
	 * Create a GPUBuffer from vertex data.
	 * @param {Float32Array|ArrayBuffer} data
	 */
	_set_gpu_vertex_buffer(data) {
		const device = this.ctx.device;

		if (this._gpu_vbos.length > 0) {
			for (const buf of this._gpu_vbos) {
				if (buf && buf.destroy) buf.destroy();
			}
		}

		const buffer = device.createBuffer({
			size: data.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true
		});

		new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer || data, data.byteOffset || 0, data.byteLength));
		buffer.unmap();

		this._gpu_vbos = [buffer];
		this.vbo = buffer;
	}

	/**
	 * Create a GPUBuffer from index data.
	 * @param {Uint16Array|Uint32Array} data
	 */
	_set_gpu_index_buffer(data) {
		const device = this.ctx.device;

		if (this._gpu_ebo)
			this._gpu_ebo.destroy();

		// WebGPU requires index buffer size to be a multiple of 4
		const aligned_size = Math.ceil(data.byteLength / 4) * 4;

		const buffer = device.createBuffer({
			size: aligned_size,
			usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true
		});

		new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
		buffer.unmap();

		this._gpu_ebo = buffer;
		this.ebo = buffer;
		this.index_count = data.length;

		if (data instanceof Uint32Array) {
			this.index_type = 'uint32';
			this._index_bytes = 4;
		} else {
			this.index_type = 'uint16';
			this._index_bytes = 2;
		}
	}

	/**
	 * Bind vertex and index buffers to a render pass encoder.
	 * @param {GPURenderPassEncoder} pass
	 */
	bind_to_pass(pass) {
		for (let i = 0; i < this._gpu_vbos.length; i++)
			pass.setVertexBuffer(i, this._gpu_vbos[i]);

		if (this._gpu_ebo)
			pass.setIndexBuffer(this._gpu_ebo, this.index_type);
	}

	/**
	 * Issue a draw call on the current render pass.
	 * @param {number} [count]
	 * @param {number} [offset=0]
	 */
	_draw_gpu(count, offset = 0) {
		const pass = this.ctx.render_pass;
		if (!pass) return;

		this.bind_to_pass(pass);

		count = count ?? this.index_count;

		if (this._gpu_ebo)
			pass.drawIndexed(count, 1, offset, 0, 0);
		else
			pass.draw(count, 1, offset, 0);
	}

	/**
	 * Get vertex buffer layouts for pipeline creation.
	 * @returns {GPUVertexBufferLayout[]}
	 */
	get vertex_layouts() {
		return this._vertex_layouts || [];
	}

	/**
	 * M2 interleaved vertex layout for WebGPU pipeline.
	 * Stride = 40 bytes: pos(3f) + normal(3f) + bone_idx(4ub) + bone_wt(4ub) + uv(2f)
	 */
	_get_m2_vertex_layout() {
		return {
			arrayStride: 40,
			attributes: [
				{ shaderLocation: AttributeLocation.POSITION, offset: 0, format: 'float32x3' },
				{ shaderLocation: AttributeLocation.NORMAL, offset: 12, format: 'float32x3' },
				{ shaderLocation: AttributeLocation.BONE_INDICES, offset: 24, format: 'uint8x4' },
				{ shaderLocation: AttributeLocation.BONE_WEIGHTS, offset: 28, format: 'unorm8x4' },
				{ shaderLocation: AttributeLocation.TEXCOORD, offset: 32, format: 'float32x2' }
			]
		};
	}

	/**
	 * M2 separate buffer vertex layouts for WebGPU pipeline.
	 */
	_get_m2_separate_vertex_layouts(has_uv2) {
		const layouts = [
			{ arrayStride: 12, attributes: [{ shaderLocation: AttributeLocation.POSITION, offset: 0, format: 'float32x3' }] },
			{ arrayStride: 12, attributes: [{ shaderLocation: AttributeLocation.NORMAL, offset: 0, format: 'float32x3' }] },
			{ arrayStride: 4, attributes: [{ shaderLocation: AttributeLocation.BONE_INDICES, offset: 0, format: 'uint8x4' }] },
			{ arrayStride: 4, attributes: [{ shaderLocation: AttributeLocation.BONE_WEIGHTS, offset: 0, format: 'unorm8x4' }] },
			{ arrayStride: 8, attributes: [{ shaderLocation: AttributeLocation.TEXCOORD, offset: 0, format: 'float32x2' }] }
		];

		if (has_uv2)
			layouts.push({ arrayStride: 8, attributes: [{ shaderLocation: AttributeLocation.TEXCOORD2, offset: 0, format: 'float32x2' }] });

		return layouts;
	}

	/**
	 * WMO separate buffer vertex layouts for WebGPU pipeline.
	 */
	_get_wmo_separate_vertex_layouts(has_color, has_color2, has_color3, has_uv2, has_uv3, has_uv4) {
		const layouts = [
			{ arrayStride: 12, attributes: [{ shaderLocation: AttributeLocation.POSITION, offset: 0, format: 'float32x3' }] },
			{ arrayStride: 12, attributes: [{ shaderLocation: AttributeLocation.NORMAL, offset: 0, format: 'float32x3' }] },
			{ arrayStride: 8, attributes: [{ shaderLocation: AttributeLocation.TEXCOORD, offset: 0, format: 'float32x2' }] }
		];

		if (has_color)
			layouts.push({ arrayStride: 4, attributes: [{ shaderLocation: AttributeLocation.COLOR, offset: 0, format: 'unorm8x4' }] });
		if (has_color2)
			layouts.push({ arrayStride: 4, attributes: [{ shaderLocation: AttributeLocation.COLOR2, offset: 0, format: 'unorm8x4' }] });
		if (has_color3)
			layouts.push({ arrayStride: 4, attributes: [{ shaderLocation: AttributeLocation.COLOR3, offset: 0, format: 'unorm8x4' }] });
		if (has_uv2)
			layouts.push({ arrayStride: 8, attributes: [{ shaderLocation: AttributeLocation.TEXCOORD2, offset: 0, format: 'float32x2' }] });
		if (has_uv3)
			layouts.push({ arrayStride: 8, attributes: [{ shaderLocation: AttributeLocation.TEXCOORD3, offset: 0, format: 'float32x2' }] });
		if (has_uv4)
			layouts.push({ arrayStride: 8, attributes: [{ shaderLocation: AttributeLocation.TEXCOORD4, offset: 0, format: 'float32x2' }] });

		return layouts;
	}
}

VertexArray.AttributeLocation = AttributeLocation;

module.exports = VertexArray;
