mod frontend;
mod hardware;
mod middle;

fn main() {
    let model = frontend::parse_onnx("data/memristor_mha_unrolled.onnx");
    println!("{:?}", model);
    println!(
        "{:?}",
        middle::tile(model.ops, &hardware::CrossbarSpec::default_128x128()),
    );
}
