// Character texture blending shader (WebGPU / WGSL)
// Translated from char.vertex.shader + char.fragment.shader

struct Uniforms {
	blend_mode: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var t_base_texture: texture_2d<f32>;
@group(1) @binding(1) var s_base_texture: sampler;
@group(1) @binding(2) var t_texture: texture_2d<f32>;
@group(1) @binding(3) var s_texture: sampler;

struct VertexInput {
	@location(0) position: vec4<f32>,
	@location(1) texcoord: vec2<f32>,
};

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) texcoord: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;
	out.position = in.position;
	out.texcoord = in.texcoord;
	return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let bm = u.blend_mode;

	if (bm == 0.0 || bm == 1.0 || bm == 9.0 || bm == 15.0) {
		// Normal / passthrough
		return textureSample(t_texture, s_texture, in.texcoord);
	} else if (bm == 4.0) {
		// MULTIPLY
		let base = textureSample(t_base_texture, s_base_texture, in.texcoord);
		let blend = textureSample(t_texture, s_texture, in.texcoord);
		return base * blend;
	} else if (bm == 7.0) {
		// SCREEN
		let base = textureSample(t_base_texture, s_base_texture, in.texcoord);
		let blend = textureSample(t_texture, s_texture, in.texcoord);
		return vec4<f32>(1.0 - (1.0 - base.rgb) * (1.0 - blend.rgb), blend.a);
	} else if (bm == 6.0) {
		// OVERLAY
		let base = textureSample(t_base_texture, s_base_texture, in.texcoord);
		let blend = textureSample(t_texture, s_texture, in.texcoord);
		let r = select(1.0 - 2.0 * (1.0 - base.r) * (1.0 - blend.r), 2.0 * base.r * blend.r, blend.r < 0.5);
		let g = select(1.0 - 2.0 * (1.0 - base.g) * (1.0 - blend.g), 2.0 * base.g * blend.g, blend.g < 0.5);
		let b = select(1.0 - 2.0 * (1.0 - base.b) * (1.0 - blend.b), 2.0 * base.b * blend.b, blend.b < 0.5);
		return vec4<f32>(r, g, b, blend.a);
	} else {
		// Unsupported mode - magenta
		return vec4<f32>(1.0, 0.0, 1.0, 1.0);
	}
}
