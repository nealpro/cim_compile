mod backend;
mod frontend;
mod hardware;
mod middle;

fn main() {
    let model = frontend::parse_onnx("data/memristor_mha_unrolled.onnx");
    let ops = middle::tile(model.ops, &hardware::CrossbarSpec::default_128x128());
    println!("{:?}", ops);
    backend::write_asm(&ops, std::path::Path::new("output.s")).expect("failed to write output.s");
    println!("wrote output.s ({} tiles)", ops.len());
}
