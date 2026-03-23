// WMO (World Map Object) shader (WebGPU / WGSL)
// Translated from wmo.vertex.shader + wmo.fragment.shader

struct Uniforms {
	view_matrix: mat4x4<f32>,
	projection_matrix: mat4x4<f32>,
	model_matrix: mat4x4<f32>,
	vertex_shader: i32,
	pixel_shader: i32,
	blend_mode: i32,
	use_vertex_color: i32,
	apply_lighting: i32,
	wireframe: i32,
	_pad0: f32,
	_pad1: f32,
	ambient_color: vec4<f32>,
	diffuse_color: vec4<f32>,
	light_dir: vec4<f32>,
	wireframe_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var t_texture1: texture_2d<f32>;
@group(1) @binding(1) var s_texture1: sampler;
@group(1) @binding(2) var t_texture2: texture_2d<f32>;
@group(1) @binding(3) var s_texture2: sampler;
@group(1) @binding(4) var t_texture3: texture_2d<f32>;
@group(1) @binding(5) var s_texture3: sampler;
@group(1) @binding(6) var t_texture4: texture_2d<f32>;
@group(1) @binding(7) var s_texture4: sampler;
@group(1) @binding(8) var t_texture5: texture_2d<f32>;
@group(1) @binding(9) var s_texture5: sampler;
@group(1) @binding(10) var t_texture6: texture_2d<f32>;
@group(1) @binding(11) var s_texture6: sampler;
@group(1) @binding(12) var t_texture7: texture_2d<f32>;
@group(1) @binding(13) var s_texture7: sampler;
@group(1) @binding(14) var t_texture8: texture_2d<f32>;
@group(1) @binding(15) var s_texture8: sampler;
@group(1) @binding(16) var t_texture9: texture_2d<f32>;
@group(1) @binding(17) var s_texture9: sampler;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) normal: vec3<f32>,
	@location(4) texcoord: vec2<f32>,
	@location(5) texcoord2: vec2<f32>,
	@location(6) color: vec4<f32>,
	@location(7) color2: vec4<f32>,
	@location(8) texcoord3: vec2<f32>,
	@location(9) texcoord4: vec2<f32>,
	@location(10) color3: vec4<f32>,
};

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) texcoord: vec2<f32>,
	@location(1) texcoord2: vec2<f32>,
	@location(2) texcoord3: vec2<f32>,
	@location(3) texcoord4: vec2<f32>,
	@location(4) normal: vec3<f32>,
	@location(5) position_view: vec3<f32>,
	@location(6) color: vec4<f32>,
	@location(7) color2: vec4<f32>,
	@location(8) color3: vec4<f32>,
};

// compute transpose(inverse(mat3(m))) for normal transformation
fn normal_matrix(m: mat4x4<f32>) -> mat3x3<f32> {
	let c0 = m[0].xyz;
	let c1 = m[1].xyz;
	let c2 = m[2].xyz;
	let det = dot(c0, cross(c1, c2));
	let inv_det = 1.0 / det;
	return mat3x3<f32>(
		cross(c1, c2) * inv_det,
		cross(c2, c0) * inv_det,
		cross(c0, c1) * inv_det
	);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;

	let world_pos = u.model_matrix * vec4<f32>(in.position, 1.0);
	let view_pos = u.view_matrix * world_pos;
	out.position = u.projection_matrix * view_pos;

	let nm = normal_matrix(u.view_matrix * u.model_matrix);
	out.normal = normalize(nm * in.normal);
	out.position_view = view_pos.xyz;

	out.color = in.color;
	out.color2 = in.color2;
	out.color3 = in.color3;

	// WMO vertex shader modes
	switch u.vertex_shader {
		default {
			out.texcoord = in.texcoord;
			out.texcoord2 = in.texcoord2;
			out.texcoord3 = in.texcoord3;
			out.texcoord4 = in.texcoord4;
		}
	}

	return out;
}

// diffuse lighting
fn calc_lighting(color: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
	if (u.apply_lighting == 0) {
		return color;
	}
	let n = normalize(normal);
	let n_dot_l = max(dot(n, normalize(-u.light_dir.xyz)), 0.0);
	let ambient = u.ambient_color.xyz * color;
	let diffuse = u.diffuse_color.xyz * color * n_dot_l;
	return ambient + diffuse;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	if (u.wireframe != 0) {
		return u.wireframe_color;
	}

	let tex1 = textureSample(t_texture1, s_texture1, in.texcoord);
	let tex2 = textureSample(t_texture2, s_texture2, in.texcoord);

	var mat_diffuse = vec3<f32>(0.0);
	var emissive = vec3<f32>(0.0);

	// WMO pixel shader modes
	switch u.pixel_shader {
		case 0 { // MapObjDiffuse
			mat_diffuse = tex1.rgb;
		}
		case 1 { // MapObjSpecular
			mat_diffuse = tex1.rgb;
		}
		case 2 { // MapObjMetal
			mat_diffuse = tex1.rgb;
		}
		case 3 { // MapObjEnv
			mat_diffuse = tex1.rgb;
			emissive = tex2.rgb * tex1.a;
		}
		case 4 { // MapObjOpaque
			mat_diffuse = tex1.rgb;
		}
		case 5 { // MapObjEnvMetal
			mat_diffuse = tex1.rgb;
			emissive = (tex1.rgb * tex1.a) * tex2.rgb;
		}
		case 6 { // MapObjTwoLayerDiffuse
			let layer1 = tex1.rgb;
			let layer2 = mix(layer1, tex2.rgb, tex2.a);
			mat_diffuse = mix(layer2, layer1, in.color2.a);
		}
		case 7 { // MapObjTwoLayerEnvMetal
			let tex3 = textureSample(t_texture3, s_texture3, in.texcoord3);
			let color_mix = mix(tex1, tex2, 1.0 - in.color2.a);
			mat_diffuse = color_mix.rgb;
			emissive = (color_mix.rgb * color_mix.a) * tex3.rgb;
		}
		case 8 { // MapObjTwoLayerTerrain
			mat_diffuse = mix(tex2.rgb, tex1.rgb, in.color2.a);
		}
		case 9 { // MapObjDiffuseEmissive
			mat_diffuse = tex1.rgb;
			emissive = tex2.rgb * tex2.a * in.color2.a;
		}
		case 10 { // MapObjMaskedEnvMetal
			let tex3 = textureSample(t_texture3, s_texture3, in.texcoord3);
			let mix_factor = clamp(tex3.a * in.color2.a, 0.0, 1.0);
			mat_diffuse = mix(mix((tex1.rgb * tex2.rgb) * 2.0, tex3.rgb, mix_factor), tex1.rgb, tex1.a);
		}
		case 11 { // MapObjEnvMetalEmissive
			let tex3 = textureSample(t_texture3, s_texture3, in.texcoord3);
			mat_diffuse = tex1.rgb;
			emissive = ((tex1.rgb * tex1.a) * tex2.rgb) + ((tex3.rgb * tex3.a) * in.color2.a);
		}
		case 12 { // MapObjTwoLayerDiffuseOpaque
			mat_diffuse = mix(tex2.rgb, tex1.rgb, in.color2.a);
		}
		case 13 { // MapObjTwoLayerDiffuseEmissive
			let t1_diffuse = tex2.rgb * (1.0 - tex2.a);
			mat_diffuse = mix(t1_diffuse, tex1.rgb, in.color2.a);
			emissive = (tex2.rgb * tex2.a) * (1.0 - in.color2.a);
		}
		case 14 { // MapObjAdditiveMaskedEnvMetal
			let tex3 = textureSample(t_texture3, s_texture3, in.texcoord3);
			mat_diffuse = mix(
				(tex1.rgb * tex2.rgb * 2.0) + (tex3.rgb * clamp(tex3.a * in.color2.a, 0.0, 1.0)),
				tex1.rgb,
				tex1.a
			);
		}
		case 15 { // MapObjTwoLayerDiffuseMod2x
			let tex3 = textureSample(t_texture3, s_texture3, in.texcoord3);
			let layer1 = tex1.rgb;
			let layer2 = mix(layer1, tex2.rgb, tex2.a);
			let layer3 = mix(layer2, layer1, in.color2.a);
			mat_diffuse = layer3 * tex3.rgb * 2.0;
		}
		case 16 { // MapObjTwoLayerDiffuseMod2xNA
			let layer1 = (tex1.rgb * tex2.rgb) * 2.0;
			mat_diffuse = mix(tex1.rgb, layer1, in.color2.a);
		}
		case 17 { // MapObjTwoLayerDiffuseAlpha
			let tex3 = textureSample(t_texture3, s_texture3, in.texcoord3);
			let layer1 = tex1.rgb;
			let layer2 = mix(layer1, tex2.rgb, tex2.a);
			let layer3 = mix(layer2, layer1, tex3.a);
			mat_diffuse = (layer3 * tex3.rgb) * 2.0;
		}
		case 18 { // MapObjLod
			mat_diffuse = tex1.rgb;
		}
		case 19 { // MapObjParallax (simplified)
			mat_diffuse = tex1.rgb;
		}
		case 20 { // MapObjUnkShader
			let tex2_20 = textureSample(t_texture2, s_texture2, in.texcoord);
			let tex3_20 = textureSample(t_texture3, s_texture3, in.texcoord2);
			let tex4_20 = textureSample(t_texture4, s_texture4, in.texcoord3);
			let tex5_20 = textureSample(t_texture5, s_texture5, in.texcoord4);
			let tex6_20 = textureSample(t_texture6, s_texture6, in.texcoord);
			let tex7_20 = textureSample(t_texture7, s_texture7, in.texcoord2);
			let tex8_20 = textureSample(t_texture8, s_texture8, in.texcoord3);
			let tex9_20 = textureSample(t_texture9, s_texture9, in.texcoord4);

			let second_color_sum = dot(in.color3.bgr, vec3<f32>(1.0));
			let weights = vec4<f32>(in.color3.bgr, 1.0 - clamp(second_color_sum, 0.0, 1.0));
			let heights = max(vec4<f32>(tex6_20.a, tex7_20.a, tex8_20.a, tex9_20.a), vec4<f32>(0.004));
			let alpha_vec = weights * heights;
			let weights_max = max(alpha_vec.r, max(alpha_vec.g, max(alpha_vec.b, alpha_vec.a)));
			let alpha_vec2 = (1.0 - clamp(vec4<f32>(weights_max) - alpha_vec, vec4<f32>(0.0), vec4<f32>(1.0))) * alpha_vec;
			let alpha_normalized = alpha_vec2 * (1.0 / dot(alpha_vec2, vec4<f32>(1.0)));

			let tex_mixed = tex2_20 * alpha_normalized.r +
				tex3_20 * alpha_normalized.g +
				tex4_20 * alpha_normalized.b +
				tex5_20 * alpha_normalized.a;

			emissive = (tex_mixed.a * tex1.rgb) * tex_mixed.rgb;
			mat_diffuse = mix(tex_mixed.rgb, vec3<f32>(0.0), in.color3.a);
		}
		default {
			mat_diffuse = tex1.rgb;
		}
	}

	// apply vertex color if enabled
	if (u.use_vertex_color != 0) {
		mat_diffuse *= in.color.rgb;
	}

	// alpha test discard (only when blend_mode > 0)
	if (u.blend_mode > 0 && tex1.a < 0.501960814) {
		discard;
	}

	// apply lighting
	var lit_color = calc_lighting(mat_diffuse, in.normal);

	// add emissive (unaffected by lighting)
	lit_color += emissive;

	return vec4<f32>(lit_color, 1.0);
}
