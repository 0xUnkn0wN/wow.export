/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const core = require('../../core');

const M3Loader = require('../loaders/M3Loader');
const Shaders = require('../Shaders');

const GLContext = require('../gl/GLContext');
const VertexArray = require('../gl/VertexArray');
const GLTexture = require('../gl/GLTexture');

const IDENTITY_MAT4 = new Float32Array([
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1
]);

class M3RendererGL {
	/**
	 * @param {BufferWrapper} data
	 * @param {GLContext} gl_context
	 * @param {boolean} [reactive=false]
	 * @param {boolean} [useRibbon=true]
	 */
	constructor(data, gl_context, reactive = false, useRibbon = true) {
		this.data = data;
		this.ctx = gl_context;
		this.gl = gl_context.gl;
		this.reactive = reactive;
		this.useRibbon = useRibbon;

		this.m3 = null;

		// rendering state
		this.vaos = [];
		this.buffers = [];
		this.draw_calls = [];
		this.default_texture = null;

		// transforms
		this.model_matrix = new Float32Array(IDENTITY_MAT4);
	}

	/**
	 * Load shader program
	 */
	static load_shaders(ctx) {
		return Shaders.create_program(ctx, 'm2');
	}

	async load() {
		this.m3 = new M3Loader(this.data);
		await this.m3.load();

		this.shader = M3RendererGL.load_shaders(this.ctx);

		if (this.ctx.is_webgpu)
			this._init_webgpu();

		this._create_default_texture();

		if (this.m3.vertices && this.m3.vertices.length > 0)
			await this.loadLOD(0);

		this.data = undefined;
	}

	_init_webgpu() {
		const shader = this.shader;

		// M2 uniform layout (M3 uses the M2 shader)
		shader.define_uniform('u_view_matrix', 0, 64);
		shader.define_uniform('u_projection_matrix', 64, 64);
		shader.define_uniform('u_model_matrix', 128, 64);
		shader.define_uniform('u_view_up', 192, 16);
		shader.define_uniform('u_time', 208, 4);
		shader.define_uniform('u_bone_count', 212, 4);
		shader.define_uniform('u_has_tex_matrix1', 216, 4);
		shader.define_uniform('u_has_tex_matrix2', 220, 4);
		shader.define_uniform('u_tex_matrix1', 224, 64);
		shader.define_uniform('u_tex_matrix2', 288, 64);
		shader.define_uniform('u_vertex_shader', 352, 4);
		shader.define_uniform('u_pixel_shader', 356, 4);
		shader.define_uniform('u_blend_mode', 360, 4);
		shader.define_uniform('u_apply_lighting', 364, 4);
		shader.define_uniform('u_mesh_color', 368, 16);
		shader.define_uniform('u_tex_sample_alpha', 384, 16);
		shader.define_uniform('u_alpha_test', 400, 4);
		shader.define_uniform('u_wireframe', 404, 4);
		shader.define_uniform('u_wireframe_color', 416, 16);
		shader.define_uniform('u_ambient_color', 432, 16);
		shader.define_uniform('u_diffuse_color', 448, 16);
		shader.define_uniform('u_light_dir', 464, 16);
		shader.define_uniform('u_bone_matrices', 480, 16384);

		shader.create_texture_bind_group_layout(4);
	}

	_create_gpu_pipeline(vao) {
		this.shader.create_pipeline({
			vertex_buffers: vao.vertex_layouts,
			topology: 'triangle-list',
			blend: {
				color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
				alpha: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' }
			},
			depth_stencil: {
				format: 'depth24plus-stencil8',
				depthWriteEnabled: true,
				depthCompare: 'less-equal'
			},
			cull_mode: 'none'
		});
	}

	_create_default_texture() {
		const pixels = new Uint8Array([87, 175, 226, 255]); // 0x57afe2 blue
		this.default_texture = new GLTexture(this.ctx);
		this.default_texture.set_rgba(pixels, 1, 1, { has_alpha: false });
	}

	async loadLOD(index) {
		this._dispose_geometry();

		const m3 = this.m3;
		const gl = this.gl;

		// build interleaved vertex buffer matching M2 format
		// format: position(3f) + normal(3f) + bone_idx(4ub) + bone_weight(4ub) + uv(2f) = 40 bytes
		const vertex_count = m3.vertices.length / 3;
		const stride = 40;
		const vertex_data = new ArrayBuffer(vertex_count * stride);
		const vertex_view = new DataView(vertex_data);

		for (let i = 0; i < vertex_count; i++) {
			const offset = i * stride;
			const v_idx = i * 3;
			const uv_idx = i * 2;

			// position
			vertex_view.setFloat32(offset + 0, m3.vertices[v_idx], true);
			vertex_view.setFloat32(offset + 4, m3.vertices[v_idx + 1], true);
			vertex_view.setFloat32(offset + 8, m3.vertices[v_idx + 2], true);

			// normal
			vertex_view.setFloat32(offset + 12, m3.normals ? m3.normals[v_idx] : 0, true);
			vertex_view.setFloat32(offset + 16, m3.normals ? m3.normals[v_idx + 1] : 1, true);
			vertex_view.setFloat32(offset + 20, m3.normals ? m3.normals[v_idx + 2] : 0, true);

			// bone indices (all zero - no skinning)
			vertex_view.setUint8(offset + 24, 0);
			vertex_view.setUint8(offset + 25, 0);
			vertex_view.setUint8(offset + 26, 0);
			vertex_view.setUint8(offset + 27, 0);

			// bone weights (first weight = 255, rest = 0 for identity transform)
			vertex_view.setUint8(offset + 28, 255);
			vertex_view.setUint8(offset + 29, 0);
			vertex_view.setUint8(offset + 30, 0);
			vertex_view.setUint8(offset + 31, 0);

			// texcoord
			vertex_view.setFloat32(offset + 32, m3.uv ? m3.uv[uv_idx] : 0, true);
			vertex_view.setFloat32(offset + 36, m3.uv ? m3.uv[uv_idx + 1] : 0, true);
		}

		// create VAO
		const vao = new VertexArray(this.ctx);
		const is_webgpu = this.ctx.is_webgpu;

		if (is_webgpu) {
			vao.set_vertex_buffer(new Uint8Array(vertex_data));
			vao.setup_m2_vertex_format();
		} else {
			vao.bind();

			const vbo = gl.createBuffer();
			gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
			gl.bufferData(gl.ARRAY_BUFFER, vertex_data, gl.STATIC_DRAW);
			this.buffers.push(vbo);
			vao.vbo = vbo;

			vao.setup_m2_vertex_format();
		}

		this.vaos.push(vao);

		// build draw calls for LOD 0 geosets
		this.draw_calls = [];

		const geosets_per_lod = m3.geosetCountPerLOD || m3.geosets.length;
		const start_geo = index * geosets_per_lod;
		const end_geo = Math.min(start_geo + geosets_per_lod, m3.geosets.length);

		for (let geo_idx = start_geo; geo_idx < end_geo; geo_idx++) {
			const geoset = m3.geosets[geo_idx];
			const indices = new Uint16Array(m3.indices.slice(geoset.indexStart, geoset.indexStart + geoset.indexCount));

			if (is_webgpu) {
				// each geoset gets its own VAO with shared VBO but separate index buffer
				const geo_vao = new VertexArray(this.ctx);
				geo_vao._gpu_vbos = vao._gpu_vbos;
				geo_vao._vertex_layouts = vao._vertex_layouts;
				geo_vao.set_index_buffer(indices);

				this.draw_calls.push({
					vao: geo_vao,
					ebo: null,
					count: geoset.indexCount,
					visible: true
				});
			} else {
				const ebo = gl.createBuffer();
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
				gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
				this.buffers.push(ebo);

				this.draw_calls.push({
					vao: vao,
					ebo: ebo,
					count: geoset.indexCount,
					visible: true
				});
			}
		}

		// create pipeline now that we have vertex layouts
		if (is_webgpu && !this.shader.pipeline)
			this._create_gpu_pipeline(vao);
	}

	updateGeosets() {
		// no geoset management for M3
	}

	updateWireframe() {
		// handled in render()
	}

	/**
	 * Get model bounding box
	 * @returns {{ min: number[], max: number[] } | null}
	 */
	getBoundingBox() {
		if (!this.m3 || !this.m3.vertices)
			return null;

		const verts = this.m3.vertices;
		const min = [Infinity, Infinity, Infinity];
		const max = [-Infinity, -Infinity, -Infinity];

		for (let i = 0; i < verts.length; i += 3) {
			min[0] = Math.min(min[0], verts[i]);
			min[1] = Math.min(min[1], verts[i + 1]);
			min[2] = Math.min(min[2], verts[i + 2]);
			max[0] = Math.max(max[0], verts[i]);
			max[1] = Math.max(max[1], verts[i + 1]);
			max[2] = Math.max(max[2], verts[i + 2]);
		}

		return { min, max };
	}

	/**
	 * Render the model
	 * @param {Float32Array} view_matrix
	 * @param {Float32Array} projection_matrix
	 */
	render(view_matrix, projection_matrix) {
		if (!this.shader || this.draw_calls.length === 0)
			return;

		const gl = this.gl;
		const ctx = this.ctx;
		const shader = this.shader;
		const wireframe = core.view.config.modelViewerWireframe;
		const is_webgpu = ctx.is_webgpu;

		shader.use();

		// scene uniforms
		shader.set_uniform_mat4('u_view_matrix', false, view_matrix);
		shader.set_uniform_mat4('u_projection_matrix', false, projection_matrix);
		shader.set_uniform_mat4('u_model_matrix', false, this.model_matrix);
		shader.set_uniform_3f('u_view_up', 0, 1, 0);
		shader.set_uniform_1f('u_time', performance.now() * 0.001);

		// set identity bone matrix for bone 0 (M3 has no skeleton)
		shader.set_uniform_1i('u_bone_count', 1);
		if (is_webgpu) {
			shader.set_uniform_mat4_array('u_bone_matrices', IDENTITY_MAT4);
		} else {
			const loc = shader.get_uniform_location('u_bone_matrices');
			if (loc !== null)
				gl.uniformMatrix4fv(loc, false, IDENTITY_MAT4);
		}

		// texture matrix defaults
		shader.set_uniform_1i('u_has_tex_matrix1', 0);
		shader.set_uniform_1i('u_has_tex_matrix2', 0);
		shader.set_uniform_mat4('u_tex_matrix1', false, IDENTITY_MAT4);
		shader.set_uniform_mat4('u_tex_matrix2', false, IDENTITY_MAT4);

		// lighting
		const lx = 3, ly = -0.7, lz = -2;
		const light_view_x = view_matrix[0] * lx + view_matrix[4] * ly + view_matrix[8] * lz;
		const light_view_y = view_matrix[1] * lx + view_matrix[5] * ly + view_matrix[9] * lz;
		const light_view_z = view_matrix[2] * lx + view_matrix[6] * ly + view_matrix[10] * lz;

		shader.set_uniform_1i('u_apply_lighting', 1);
		shader.set_uniform_3f('u_ambient_color', 0.5, 0.5, 0.5);
		shader.set_uniform_3f('u_diffuse_color', 0.7, 0.7, 0.7);
		shader.set_uniform_3f('u_light_dir', light_view_x, light_view_y, light_view_z);

		// wireframe
		shader.set_uniform_1i('u_wireframe', wireframe ? 1 : 0);
		shader.set_uniform_4f('u_wireframe_color', 1, 1, 1, 1);

		// alpha test
		shader.set_uniform_1f('u_alpha_test', 0.501960814);

		// material settings (opaque)
		shader.set_uniform_1i('u_vertex_shader', 0);
		shader.set_uniform_1i('u_pixel_shader', 0);
		shader.set_uniform_1i('u_blend_mode', 0);
		shader.set_uniform_4f('u_mesh_color', 1, 1, 1, 1);
		shader.set_uniform_3f('u_tex_sample_alpha', 1, 1, 1);

		if (is_webgpu) {
			const pass = ctx.render_pass;
			if (!pass || !shader.pipeline)
				return;

			pass.setPipeline(shader.pipeline);

			// create texture bind group with default textures
			const tex_objs = [this.default_texture, this.default_texture, this.default_texture, this.default_texture];
			let tex_bind_group;
			try {
				tex_bind_group = shader.create_texture_bind_group(tex_objs);
			} catch (e) {
				return;
			}

			for (const dc of this.draw_calls) {
				if (!dc.visible)
					continue;

				shader.flush_uniforms();
				pass.setBindGroup(0, shader.uniform_bind_group);
				pass.setBindGroup(1, tex_bind_group);

				dc.vao.bind_to_pass(pass);
				pass.drawIndexed(dc.count, 1, 0, 0, 0);
			}
		} else {
			// texture samplers (WebGL only)
			shader.set_uniform_1i('u_texture1', 0);
			shader.set_uniform_1i('u_texture2', 1);
			shader.set_uniform_1i('u_texture3', 2);
			shader.set_uniform_1i('u_texture4', 3);

			this.default_texture.bind(0);
			this.default_texture.bind(1);
			this.default_texture.bind(2);
			this.default_texture.bind(3);

			ctx.set_blend(false);
			ctx.set_depth_test(true);
			ctx.set_cull_face(false);

			for (const dc of this.draw_calls) {
				if (!dc.visible)
					continue;

				dc.vao.bind();
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, dc.ebo);
				gl.drawElements(
					wireframe ? gl.LINES : gl.TRIANGLES,
					dc.count,
					gl.UNSIGNED_SHORT,
					0
				);
			}

			ctx.set_cull_face(false);
		}
	}

	_dispose_geometry() {
		for (const vao of this.vaos)
			vao.dispose();

		for (const buf of this.buffers)
			this.gl.deleteBuffer(buf);

		this.vaos = [];
		this.buffers = [];
		this.draw_calls = [];
	}

	dispose() {
		this._dispose_geometry();

		if (this.default_texture) {
			this.default_texture.dispose();
			this.default_texture = null;
		}
	}
}

module.exports = M3RendererGL;
