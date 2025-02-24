
// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};


@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.normal = model.normal;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0); // 2.
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let up = in.normal.y * 0.5 + 0.5;
    let color = vec3<f32>(0.0, 1.0, 0.0);
    let ambient_down = vec3<f32>(0.0, 0.0, 0.0);
    let ambient_up = vec3<f32>(1.0, 1.0, 1.0);
    let ambient = ambient_down + ambient_up * up;
    return vec4<f32>(color * ambient, 1.0);
}
