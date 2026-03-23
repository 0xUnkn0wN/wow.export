/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const BlendMode = {
	OPAQUE: 0,
	ALPHA_KEY: 1,
	ALPHA: 2,
	ADD: 3,
	MOD: 4,
	MOD2X: 5,
	MOD_ADD: 6,
	INV_SRC_ALPHA_ADD: 7,
	INV_SRC_ALPHA_OPAQUE: 8,
	SRC_ALPHA_OPAQUE: 9,
	NO_ALPHA_ADD: 10,
	CONSTANT_ALPHA: 11,
	SCREEN: 12,
	BLEND_ADD: 13
};

// Maps WebGL blend factor constants to WebGPU blend factor strings.
const BLEND_FACTOR_MAP = {
	0: 'zero',
	1: 'one',
	0x0300: 'src-alpha',
	0x0301: 'one-minus-src-alpha',
	0x0302: 'dst-alpha',
	0x0303: 'one-minus-dst-alpha',
	0x0304: 'dst-color',     // GL_DST_COLOR
	0x0306: 'src-color',     // GL_SRC_COLOR
	0x8001: 'constant-alpha', // GL_CONSTANT_ALPHA
	0x8002: 'one-minus-constant-alpha' // GL_ONE_MINUS_CONSTANT_ALPHA
};

// Maps WebGL compare function constants to WebGPU compare function strings.
const COMPARE_FUNC_MAP = {
	0x0200: 'never',
	0x0201: 'less',
	0x0202: 'equal',
	0x0203: 'less-equal',
	0x0204: 'greater',
	0x0205: 'not-equal',
	0x0206: 'greater-equal',
	0x0207: 'always'
};

class GPUContext {
	/**
	 * @param {HTMLCanvasElement} canvas
	 * @param {object} [options]
	 */
	constructor(canvas, options = {}) {
		this.canvas = canvas;
		this.options = options;

		this.device = null;
		this.gpu_context = null;
		this.format = null;
		this.depth_texture = null;
		this.msaa_texture = null;
		this.sample_count = options.antialias !== false ? 4 : 1;

		// state cache (mirrors GLContext interface)
		this._depth_test = true;
		this._depth_write = true;
		this._depth_func = 0x0203; // GL_LEQUAL
		this._cull_face = false;
		this._cull_mode = 0x0405; // GL_BACK
		this._blend = false;
		this._blend_src = 1; // GL_ONE
		this._blend_dst = 0; // GL_ZERO
		this._blend_src_alpha = 1;
		this._blend_dst_alpha = 0;
		this._current_program = null;
		this._current_vao = null;
		this._bound_textures = new Array(16).fill(null);
		this._active_texture_unit = 0;

		this._clear_color = { r: 0, g: 0, b: 0, a: 0 };

		this.viewport_width = 0;
		this.viewport_height = 0;

		// WebGL constant compatibility layer for renderers that reference gl.* constants
		this.gl = this._create_gl_compat();

		this._ready = false;
	}

	/**
	 * Async initialization - must be called after construction.
	 * @returns {Promise<GPUContext>}
	 */
	async init() {
		if (!navigator.gpu)
			throw new Error('WebGPU not supported');

		const adapter = await navigator.gpu.requestAdapter({
			powerPreference: 'high-performance'
		});

		if (!adapter)
			throw new Error('WebGPU adapter not available');

		this.adapter = adapter;
		this.device = await adapter.requestDevice();

		this.device.lost.then(info => {
			console.error('WebGPU device lost:', info.message);
		});

		this.gpu_context = this.canvas.getContext('webgpu');
		this.format = navigator.gpu.getPreferredCanvasFormat();

		this.gpu_context.configure({
			device: this.device,
			format: this.format,
			alphaMode: this.options.alpha !== false ? 'premultiplied' : 'opaque'
		});

		this._create_depth_texture();
		this._ready = true;

		// extension compat (report capabilities)
		this.ext_s3tc = adapter.features.has('texture-compression-bc');
		this.ext_s3tc_srgb = adapter.features.has('texture-compression-bc');
		this.ext_aniso = true; // WebGPU has anisotropic filtering built-in
		this.max_anisotropy = 16;
		this.ext_float_texture = true; // WebGPU supports float textures natively

		return this;
	}

	_create_depth_texture() {
		const width = Math.max(1, this.canvas.width);
		const height = Math.max(1, this.canvas.height);

		if (this.depth_texture)
			this.depth_texture.destroy();

		this.depth_texture = this.device.createTexture({
			size: [width, height],
			format: 'depth24plus-stencil8',
			usage: GPUTextureUsage.RENDER_ATTACHMENT,
			sampleCount: this.sample_count
		});

		if (this.sample_count > 1) {
			if (this.msaa_texture)
				this.msaa_texture.destroy();

			this.msaa_texture = this.device.createTexture({
				size: [width, height],
				format: this.format,
				usage: GPUTextureUsage.RENDER_ATTACHMENT,
				sampleCount: this.sample_count
			});
		}
	}

	/**
	 * Creates a compatibility object that provides WebGL constants.
	 * Renderers use this.gl.TRIANGLES, this.gl.FLOAT, etc.
	 */
	_create_gl_compat() {
		return {
			// primitive types
			POINTS: 0x0000,
			LINES: 0x0001,
			LINE_LOOP: 0x0002,
			LINE_STRIP: 0x0003,
			TRIANGLES: 0x0004,
			TRIANGLE_STRIP: 0x0005,
			TRIANGLE_FAN: 0x0006,

			// data types
			BYTE: 0x1400,
			UNSIGNED_BYTE: 0x1401,
			SHORT: 0x1402,
			UNSIGNED_SHORT: 0x1403,
			INT: 0x1404,
			UNSIGNED_INT: 0x1405,
			FLOAT: 0x1406,

			// buffer targets
			ARRAY_BUFFER: 0x8892,
			ELEMENT_ARRAY_BUFFER: 0x8893,
			UNIFORM_BUFFER: 0x8A11,

			// buffer usage
			STATIC_DRAW: 0x88E4,
			DYNAMIC_DRAW: 0x88E8,
			STREAM_DRAW: 0x88E0,

			// texture targets
			TEXTURE_2D: 0x0DE1,
			TEXTURE_CUBE_MAP: 0x8513,

			// texture units
			TEXTURE0: 0x84C0,

			// texture params
			TEXTURE_MAG_FILTER: 0x2800,
			TEXTURE_MIN_FILTER: 0x2801,
			TEXTURE_WRAP_S: 0x2802,
			TEXTURE_WRAP_T: 0x2803,
			NEAREST: 0x2600,
			LINEAR: 0x2601,
			NEAREST_MIPMAP_NEAREST: 0x2700,
			LINEAR_MIPMAP_NEAREST: 0x2701,
			NEAREST_MIPMAP_LINEAR: 0x2702,
			LINEAR_MIPMAP_LINEAR: 0x2703,
			CLAMP_TO_EDGE: 0x812F,
			REPEAT: 0x2901,

			// pixel formats
			RGBA: 0x1908,
			RGB: 0x1907,
			RGBA8: 0x8058,

			// depth/stencil
			DEPTH_TEST: 0x0B71,
			DEPTH_BUFFER_BIT: 0x00000100,
			COLOR_BUFFER_BIT: 0x00004000,
			STENCIL_BUFFER_BIT: 0x00000400,

			// depth functions
			NEVER: 0x0200,
			LESS: 0x0201,
			EQUAL: 0x0202,
			LEQUAL: 0x0203,
			GREATER: 0x0204,
			NOTEQUAL: 0x0205,
			GEQUAL: 0x0206,
			ALWAYS: 0x0207,

			// culling
			CULL_FACE: 0x0B44,
			FRONT: 0x0404,
			BACK: 0x0405,
			FRONT_AND_BACK: 0x0408,

			// blending
			BLEND: 0x0BE2,
			ZERO: 0,
			ONE: 1,
			SRC_COLOR: 0x0300,
			ONE_MINUS_SRC_COLOR: 0x0301,
			SRC_ALPHA: 0x0302,
			ONE_MINUS_SRC_ALPHA: 0x0303,
			DST_ALPHA: 0x0304,
			ONE_MINUS_DST_ALPHA: 0x0305,
			DST_COLOR: 0x0306,
			ONE_MINUS_DST_COLOR: 0x0307,
			CONSTANT_ALPHA: 0x8003,
			ONE_MINUS_CONSTANT_ALPHA: 0x8004,

			// shader types
			VERTEX_SHADER: 0x8B31,
			FRAGMENT_SHADER: 0x8B30,

			// program params
			LINK_STATUS: 0x8B82,
			COMPILE_STATUS: 0x8B81,
			INVALID_INDEX: 0xFFFFFFFF
		};
	}

	/**
	 * Begins a new render pass. Call before issuing draw commands.
	 * @returns {{ encoder: GPUCommandEncoder, pass: GPURenderPassEncoder }}
	 */
	begin_render_pass() {
		const encoder = this.device.createCommandEncoder();
		const texture_view = this.gpu_context.getCurrentTexture().createView();

		const color_attachment = {
			view: this.sample_count > 1 ? this.msaa_texture.createView() : texture_view,
			loadOp: 'clear',
			storeOp: 'store',
			clearValue: this._clear_color
		};

		if (this.sample_count > 1)
			color_attachment.resolveTarget = texture_view;

		const pass = encoder.beginRenderPass({
			colorAttachments: [color_attachment],
			depthStencilAttachment: {
				view: this.depth_texture.createView(),
				depthClearValue: 1.0,
				depthLoadOp: 'clear',
				depthStoreOp: 'store',
				stencilClearValue: 0,
				stencilLoadOp: 'clear',
				stencilStoreOp: 'store'
			}
		});

		return { encoder, pass };
	}

	/**
	 * Ends a render pass and submits the command buffer.
	 * @param {{ encoder: GPUCommandEncoder, pass: GPURenderPassEncoder }} render_state
	 */
	end_render_pass(render_state) {
		render_state.pass.end();
		this.device.queue.submit([render_state.encoder.finish()]);
	}

	/**
	 * Begin a frame. Creates a render pass that all renderers share.
	 * Call at the start of the render loop, before any renderer.render().
	 */
	begin_frame() {
		const state = this.begin_render_pass();
		this._frame_encoder = state.encoder;
		this._frame_pass = state.pass;
		this.render_pass = state.pass;

		// set viewport on the render pass
		if (this.viewport_width && this.viewport_height)
			this._frame_pass.setViewport(0, 0, this.viewport_width, this.viewport_height, 0, 1);
	}

	/**
	 * End the current frame. Submits all commands.
	 * Call after all renderers have rendered.
	 */
	end_frame() {
		if (this._frame_pass) {
			this._frame_pass.end();
			this.device.queue.submit([this._frame_encoder.finish()]);
			this._frame_pass = null;
			this._frame_encoder = null;
			this.render_pass = null;
		}
	}

	/**
	 * @param {number} width
	 * @param {number} height
	 */
	set_viewport(width, height) {
		this.viewport_width = width;
		this.viewport_height = height;

		if (this.canvas.width !== width || this.canvas.height !== height) {
			this.canvas.width = width;
			this.canvas.height = height;

			if (this._ready)
				this._create_depth_texture();
		}
	}

	/**
	 * @param {boolean} color
	 * @param {boolean} depth
	 * @param {boolean} stencil
	 */
	clear(color = true, depth = true, stencil = false) {
		// In WebGPU, clearing is done as part of the render pass (loadOp: 'clear').
		// This method is kept for API compatibility. The begin_render_pass() method
		// uses the stored clear color and always clears. For selective clearing,
		// renderers should configure their render passes directly.
	}

	/**
	 * @param {number} r
	 * @param {number} g
	 * @param {number} b
	 * @param {number} a
	 */
	set_clear_color(r, g, b, a = 1) {
		this._clear_color = { r, g, b, a };
	}

	/**
	 * @param {boolean} enable
	 */
	set_depth_test(enable) {
		this._depth_test = enable;
	}

	/**
	 * @param {boolean} enable
	 */
	set_depth_write(enable) {
		this._depth_write = enable;
	}

	/**
	 * @param {number} func - WebGL depth function constant
	 */
	set_depth_func(func) {
		this._depth_func = func;
	}

	/**
	 * @param {boolean} enable
	 */
	set_cull_face(enable) {
		this._cull_face = enable;
	}

	/**
	 * @param {number} mode - gl.FRONT, gl.BACK, or gl.FRONT_AND_BACK
	 */
	set_cull_mode(mode) {
		this._cull_mode = mode;
	}

	/**
	 * @param {boolean} enable
	 */
	set_blend(enable) {
		this._blend = enable;
	}

	/**
	 * @param {number} src
	 * @param {number} dst
	 */
	set_blend_func(src, dst) {
		this._blend_src = src;
		this._blend_dst = dst;
		this._blend_src_alpha = src;
		this._blend_dst_alpha = dst;
	}

	/**
	 * @param {number} src_rgb
	 * @param {number} dst_rgb
	 * @param {number} src_alpha
	 * @param {number} dst_alpha
	 */
	set_blend_func_separate(src_rgb, dst_rgb, src_alpha, dst_alpha) {
		this._blend_src = src_rgb;
		this._blend_dst = dst_rgb;
		this._blend_src_alpha = src_alpha;
		this._blend_dst_alpha = dst_alpha;
	}

	/**
	 * Apply WoW blend mode (mirrors GLContext)
	 * @param {number} blend_mode
	 */
	apply_blend_mode(blend_mode) {
		const gl = this.gl;

		switch (blend_mode) {
			case BlendMode.OPAQUE:
				this.set_blend(false);
				this.set_depth_write(true);
				break;

			case BlendMode.ALPHA_KEY:
				this.set_blend(true);
				this.set_blend_func(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
				this.set_depth_write(true);
				break;

			case BlendMode.ALPHA:
				this.set_blend(true);
				this.set_blend_func(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
				this.set_depth_write(false);
				break;

			case BlendMode.ADD:
			case BlendMode.NO_ALPHA_ADD:
				this.set_blend(true);
				this.set_blend_func(gl.SRC_ALPHA, gl.ONE);
				this.set_depth_write(false);
				break;

			case BlendMode.MOD:
				this.set_blend(true);
				this.set_blend_func(gl.DST_COLOR, gl.ZERO);
				this.set_depth_write(false);
				break;

			case BlendMode.MOD2X:
				this.set_blend(true);
				this.set_blend_func(gl.DST_COLOR, gl.SRC_COLOR);
				this.set_depth_write(false);
				break;

			case BlendMode.MOD_ADD:
				this.set_blend(true);
				this.set_blend_func_separate(gl.DST_COLOR, gl.ONE, gl.DST_ALPHA, gl.ONE);
				this.set_depth_write(false);
				break;

			case BlendMode.INV_SRC_ALPHA_ADD:
				this.set_blend(true);
				this.set_blend_func(gl.ONE_MINUS_SRC_ALPHA, gl.ONE);
				this.set_depth_write(false);
				break;

			case BlendMode.INV_SRC_ALPHA_OPAQUE:
				this.set_blend(true);
				this.set_blend_func(gl.ONE_MINUS_SRC_ALPHA, gl.ZERO);
				this.set_depth_write(true);
				break;

			case BlendMode.SRC_ALPHA_OPAQUE:
				this.set_blend(true);
				this.set_blend_func(gl.SRC_ALPHA, gl.ZERO);
				this.set_depth_write(true);
				break;

			case BlendMode.CONSTANT_ALPHA:
				this.set_blend(true);
				this.set_blend_func(gl.CONSTANT_ALPHA, gl.ONE_MINUS_CONSTANT_ALPHA);
				this.set_depth_write(false);
				break;

			case BlendMode.SCREEN:
				this.set_blend(true);
				this.set_blend_func(gl.ONE, gl.ONE_MINUS_SRC_COLOR);
				this.set_depth_write(false);
				break;

			case BlendMode.BLEND_ADD:
				this.set_blend(true);
				this.set_blend_func_separate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
				this.set_depth_write(false);
				break;

			default:
				this.set_blend(false);
				this.set_depth_write(true);
		}
	}

	/**
	 * Returns the current depth-stencil state descriptor for pipeline creation.
	 * @returns {object}
	 */
	get_depth_stencil_state() {
		return {
			format: 'depth24plus-stencil8',
			depthWriteEnabled: this._depth_write,
			depthCompare: this._depth_test ? (COMPARE_FUNC_MAP[this._depth_func] || 'less-equal') : 'always'
		};
	}

	/**
	 * Returns the current blend state descriptor for pipeline creation.
	 * @returns {object|undefined}
	 */
	get_blend_state() {
		if (!this._blend)
			return undefined;

		return {
			color: {
				srcFactor: BLEND_FACTOR_MAP[this._blend_src] || 'one',
				dstFactor: BLEND_FACTOR_MAP[this._blend_dst] || 'zero',
				operation: 'add'
			},
			alpha: {
				srcFactor: BLEND_FACTOR_MAP[this._blend_src_alpha] || 'one',
				dstFactor: BLEND_FACTOR_MAP[this._blend_dst_alpha] || 'zero',
				operation: 'add'
			}
		};
	}

	/**
	 * Returns the current cull mode for pipeline creation.
	 * @returns {string}
	 */
	get_cull_mode() {
		if (!this._cull_face)
			return 'none';

		if (this._cull_mode === 0x0404) // GL_FRONT
			return 'front';

		return 'back';
	}

	/**
	 * @param {object} program - pipeline or program object
	 */
	use_program(program) {
		this._current_program = program;
	}

	/**
	 * @param {object} vao
	 */
	bind_vao(vao) {
		this._current_vao = vao;
	}

	/**
	 * @param {number} unit
	 */
	active_texture(unit) {
		this._active_texture_unit = unit;
	}

	/**
	 * @param {number} unit
	 * @param {object} texture
	 * @param {number} [target]
	 */
	bind_texture(unit, texture, target) {
		this._bound_textures[unit] = texture;
		this._active_texture_unit = unit;
	}

	/**
	 * @param {number} mode
	 * @param {number} count
	 * @param {number} type
	 * @param {number} offset
	 */
	draw_elements(mode, count, type, offset) {
		// In WebGPU, draw calls are issued on the render pass encoder.
		// Renderers using WebGPU should call pass.drawIndexed() directly.
	}

	/**
	 * @param {number} mode
	 * @param {number} first
	 * @param {number} count
	 */
	draw_arrays(mode, first, count) {
		// In WebGPU, draw calls are issued on the render pass encoder.
		// Renderers using WebGPU should call pass.draw() directly.
	}

	/**
	 * Whether this context is a WebGPU context.
	 * @returns {boolean}
	 */
	get is_webgpu() {
		return true;
	}

	dispose() {
		if (this.depth_texture) {
			this.depth_texture.destroy();
			this.depth_texture = null;
		}

		if (this.msaa_texture) {
			this.msaa_texture.destroy();
			this.msaa_texture = null;
		}

		if (this.device) {
			this.device.destroy();
			this.device = null;
		}

		this.gpu_context = null;
		this.canvas = null;
	}
}

GPUContext.BlendMode = BlendMode;

module.exports = GPUContext;
