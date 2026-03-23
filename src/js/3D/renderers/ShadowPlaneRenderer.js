/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const ShaderProgram = require('../gl/ShaderProgram');

const SHADOW_VERT_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_uv;

uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;

out vec2 v_uv;

void main() {
	gl_Position = u_projection_matrix * u_view_matrix * vec4(a_position, 1.0);
	v_uv = a_uv;
}
`;

const SHADOW_FRAG_SHADER = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 frag_color;

uniform float u_shadow_radius;

void main() {
	vec2 center = vec2(0.5, 0.5);
	float dist = distance(v_uv, center) * 2.0;
	float alpha = smoothstep(1.0, 0.0, dist / (u_shadow_radius / 10.0));
	frag_color = vec4(0.0, 0.0, 0.0, alpha * 0.6);
}
`;

const SHADOW_WGSL = `
struct Uniforms {
	view_matrix: mat4x4<f32>,
	projection_matrix: mat4x4<f32>,
	shadow_radius: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) uv: vec2<f32>,
};

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;
	out.position = u.projection_matrix * u.view_matrix * vec4<f32>(in.position, 1.0);
	out.uv = in.uv;
	return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let center = vec2<f32>(0.5, 0.5);
	let dist = distance(in.uv, center) * 2.0;
	let alpha = smoothstep(1.0, 0.0, dist / (u.shadow_radius / 10.0));
	return vec4<f32>(0.0, 0.0, 0.0, alpha * 0.6);
}
`;

class ShadowPlaneRenderer {
	/**
	 * @param {GLContext|GPUContext} gl_context
	 * @param {number} size - size of the shadow plane
	 */
	constructor(gl_context, size = 2) {
		this.ctx = gl_context;
		this.gl = gl_context.gl;
		this.size = size;
		this.shadow_radius = 8.0;
		this.visible = true;

		this.shader = null;
		this.vao = null;
		this.vertex_buffer = null;
		this.index_buffer = null;

		// WebGPU resources
		this._gpu_vbo = null;
		this._gpu_ebo = null;
		this._gpu_pipeline = null;

		this._init();
	}

	_init() {
		this._create_shader();
		this._create_geometry();

		if (this.ctx.is_webgpu)
			this._create_gpu_pipeline();
	}

	_create_shader() {
		if (this.ctx.is_webgpu) {
			this.shader = new ShaderProgram(this.ctx, SHADOW_WGSL);
			// mat4 + mat4 + f32 (with padding to 16-byte alignment)
			this.shader.define_uniform('u_view_matrix', 0, 64);
			this.shader.define_uniform('u_projection_matrix', 64, 64);
			this.shader.define_uniform('u_shadow_radius', 128, 4);
		} else {
			this.shader = new ShaderProgram(this.ctx, SHADOW_VERT_SHADER, SHADOW_FRAG_SHADER);
		}
	}

	_create_geometry() {
		const half = this.size / 2;

		// quad vertices: position (xyz) + uv (st)
		const vertices = new Float32Array([
			-half, 0, -half, 0, 0,
			 half, 0, -half, 1, 0,
			 half, 0,  half, 1, 1,
			-half, 0,  half, 0, 1
		]);

		const indices = new Uint16Array([
			0, 1, 2,
			0, 2, 3
		]);

		if (this.ctx.is_webgpu) {
			const device = this.ctx.device;

			this._gpu_vbo = device.createBuffer({
				size: vertices.byteLength,
				usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
				mappedAtCreation: true
			});
			new Float32Array(this._gpu_vbo.getMappedRange()).set(vertices);
			this._gpu_vbo.unmap();

			// WebGPU requires index buffer size to be multiple of 4
			this._gpu_ebo = device.createBuffer({
				size: Math.ceil(indices.byteLength / 4) * 4,
				usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
				mappedAtCreation: true
			});
			new Uint16Array(this._gpu_ebo.getMappedRange()).set(indices);
			this._gpu_ebo.unmap();
		} else {
			const gl = this.gl;

			// create VAO
			this.vao = gl.createVertexArray();
			gl.bindVertexArray(this.vao);

			// vertex buffer
			this.vertex_buffer = gl.createBuffer();
			gl.bindBuffer(gl.ARRAY_BUFFER, this.vertex_buffer);
			gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

			// index buffer
			this.index_buffer = gl.createBuffer();
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.index_buffer);
			gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

			// position attribute (location 0)
			gl.enableVertexAttribArray(0);
			gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);

			// uv attribute (location 1)
			gl.enableVertexAttribArray(1);
			gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);

			gl.bindVertexArray(null);
		}
	}

	_create_gpu_pipeline() {
		this.shader.create_pipeline({
			vertex_buffers: [{
				arrayStride: 20,
				attributes: [
					{ shaderLocation: 0, offset: 0, format: 'float32x3' },
					{ shaderLocation: 1, offset: 12, format: 'float32x2' }
				]
			}],
			topology: 'triangle-list',
			blend: {
				color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
				alpha: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' }
			},
			depth_stencil: {
				format: 'depth24plus-stencil8',
				depthWriteEnabled: false,
				depthCompare: 'less-equal'
			}
		});
	}

	/**
	 * @param {Float32Array} view_matrix
	 * @param {Float32Array} projection_matrix
	 */
	render(view_matrix, projection_matrix) {
		if (!this.visible || !this.shader || !this.shader.is_valid())
			return;

		if (this.ctx.is_webgpu) {
			this._render_gpu(view_matrix, projection_matrix);
			return;
		}

		const gl = this.gl;

		// enable blending for transparency
		gl.enable(gl.BLEND);
		gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
		gl.depthMask(false);

		this.shader.use();
		this.shader.set_uniform_mat4('u_view_matrix', false, view_matrix);
		this.shader.set_uniform_mat4('u_projection_matrix', false, projection_matrix);
		this.shader.set_uniform_1f('u_shadow_radius', this.shadow_radius);

		this.ctx.bind_vao(this.vao);
		gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

		// restore state
		gl.depthMask(true);
		gl.disable(gl.BLEND);
	}

	_render_gpu(view_matrix, projection_matrix) {
		const pass = this.ctx.render_pass;
		if (!pass || !this.shader.pipeline) return;

		this.shader.set_uniform_mat4('u_view_matrix', false, view_matrix);
		this.shader.set_uniform_mat4('u_projection_matrix', false, projection_matrix);
		this.shader.set_uniform_1f('u_shadow_radius', this.shadow_radius);
		this.shader.flush_uniforms();

		pass.setPipeline(this.shader.pipeline);
		pass.setBindGroup(0, this.shader.uniform_bind_group);
		pass.setVertexBuffer(0, this._gpu_vbo);
		pass.setIndexBuffer(this._gpu_ebo, 'uint16');
		pass.drawIndexed(6, 1, 0, 0, 0);
	}

	dispose() {
		if (this.ctx.is_webgpu) {
			if (this._gpu_vbo) { this._gpu_vbo.destroy(); this._gpu_vbo = null; }
			if (this._gpu_ebo) { this._gpu_ebo.destroy(); this._gpu_ebo = null; }
			if (this.shader) { this.shader.dispose(); this.shader = null; }
			return;
		}

		const gl = this.gl;

		if (this.vao) {
			gl.deleteVertexArray(this.vao);
			this.vao = null;
		}

		if (this.vertex_buffer) {
			gl.deleteBuffer(this.vertex_buffer);
			this.vertex_buffer = null;
		}

		if (this.index_buffer) {
			gl.deleteBuffer(this.index_buffer);
			this.index_buffer = null;
		}

		if (this.shader) {
			this.shader.dispose();
			this.shader = null;
		}
	}
}

module.exports = ShadowPlaneRenderer;
