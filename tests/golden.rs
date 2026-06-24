use cim_compile::CompileConfig;
use cim_compile::cim::{ProjectionKind, parse_program};
use cim_compile::ir::{NormalizedOp, NormalizedProgram, ProjectionOp};
use cim_compile::lowering::lower_program;
use cim_compile::memtorch::{build_package, manifest_json};

fn tiny_program() -> NormalizedProgram {
    NormalizedProgram::new(
        "tiny",
        vec![NormalizedOp::Projection(
            ProjectionOp::new(
                "main",
                ProjectionKind::Main,
                2,
                2,
                vec![-1.0, 0.0, 0.5, 1.0],
                None,
            )
            .unwrap(),
        )],
    )
}

#[test]
fn golden_cim_text_for_tiny_projection() {
    let lowered = lower_program(&tiny_program(), CompileConfig::square(1, 8)).unwrap();
    let expected = concat!(
        "cim.module @cim_compile {\n",
        "  cim.func @main {\n",
        "    cim.tile.dispatch { projection = \"main\", tile = [0, 0], matrix_shape = [2, 2], tile_size = [1, 1], weight_offset = 0, quant_scale = 0.007874016, order = 0 }\n",
        "    cim.tile.dispatch { projection = \"main\", tile = [0, 1], matrix_shape = [2, 2], tile_size = [1, 1], weight_offset = 1, quant_scale = 1.0, order = 1 }\n",
        "    cim.tile.dispatch { projection = \"main\", tile = [1, 0], matrix_shape = [2, 2], tile_size = [1, 1], weight_offset = 2, quant_scale = 0.003937008, order = 2 }\n",
        "    cim.tile.dispatch { projection = \"main\", tile = [1, 1], matrix_shape = [2, 2], tile_size = [1, 1], weight_offset = 3, quant_scale = 0.007874016, order = 3 }\n",
        "  }\n",
        "}\n"
    );

    assert_eq!(lowered.program.to_text(), expected);
    assert_eq!(parse_program(expected).unwrap(), lowered.program);
}

#[test]
fn golden_memtorch_manifest_for_tiny_projection() {
    let normalized = tiny_program();
    let lowered = lower_program(&normalized, CompileConfig::square(1, 8)).unwrap();
    let package = build_package(&normalized, &lowered).unwrap();
    let expected = concat!(
        "{\n",
        "  \"schema_version\": 1,\n",
        "  \"entry\": \"tiny\",\n",
        "  \"tile_size\": [\n",
        "    1,\n",
        "    1\n",
        "  ],\n",
        "  \"quant_bits\": 8,\n",
        "  \"weights_file\": \"memtorch_weights.bin\",\n",
        "  \"projections\": [\n",
        "    {\n",
        "      \"name\": \"main\",\n",
        "      \"projection\": \"main\",\n",
        "      \"stage\": \"projection\",\n",
        "      \"parent\": null,\n",
        "      \"target\": \"cim\",\n",
        "      \"rows\": 2,\n",
        "      \"cols\": 2,\n",
        "      \"bias\": null,\n",
        "      \"tiles\": [\n",
        "        {\n",
        "          \"row\": 0,\n",
        "          \"col\": 0,\n",
        "          \"stage\": \"projection\",\n",
        "          \"parent\": null,\n",
        "          \"target\": \"cim\",\n",
        "          \"reason\": \"static-weight projection `main` is CiM-friendly and maps cleanly to tiles\",\n",
        "          \"matrix_shape\": [\n",
        "            2,\n",
        "            2\n",
        "          ],\n",
        "          \"tile_size\": [\n",
        "            1,\n",
        "            1\n",
        "          ],\n",
        "          \"weight_offset\": 0,\n",
        "          \"quant_scale\": 0.007874016,\n",
        "          \"order\": 0\n",
        "        },\n",
        "        {\n",
        "          \"row\": 0,\n",
        "          \"col\": 1,\n",
        "          \"stage\": \"projection\",\n",
        "          \"parent\": null,\n",
        "          \"target\": \"cim\",\n",
        "          \"reason\": \"static-weight projection `main` is CiM-friendly and maps cleanly to tiles\",\n",
        "          \"matrix_shape\": [\n",
        "            2,\n",
        "            2\n",
        "          ],\n",
        "          \"tile_size\": [\n",
        "            1,\n",
        "            1\n",
        "          ],\n",
        "          \"weight_offset\": 1,\n",
        "          \"quant_scale\": 1.0,\n",
        "          \"order\": 1\n",
        "        },\n",
        "        {\n",
        "          \"row\": 1,\n",
        "          \"col\": 0,\n",
        "          \"stage\": \"projection\",\n",
        "          \"parent\": null,\n",
        "          \"target\": \"cim\",\n",
        "          \"reason\": \"static-weight projection `main` is CiM-friendly and maps cleanly to tiles\",\n",
        "          \"matrix_shape\": [\n",
        "            2,\n",
        "            2\n",
        "          ],\n",
        "          \"tile_size\": [\n",
        "            1,\n",
        "            1\n",
        "          ],\n",
        "          \"weight_offset\": 2,\n",
        "          \"quant_scale\": 0.003937008,\n",
        "          \"order\": 2\n",
        "        },\n",
        "        {\n",
        "          \"row\": 1,\n",
        "          \"col\": 1,\n",
        "          \"stage\": \"projection\",\n",
        "          \"parent\": null,\n",
        "          \"target\": \"cim\",\n",
        "          \"reason\": \"static-weight projection `main` is CiM-friendly and maps cleanly to tiles\",\n",
        "          \"matrix_shape\": [\n",
        "            2,\n",
        "            2\n",
        "          ],\n",
        "          \"tile_size\": [\n",
        "            1,\n",
        "            1\n",
        "          ],\n",
        "          \"weight_offset\": 3,\n",
        "          \"quant_scale\": 0.007874016,\n",
        "          \"order\": 3\n",
        "        }\n",
        "      ]\n",
        "    }\n",
        "  ],\n",
        "  \"execution_plan\": [\n",
        "    {\n",
        "      \"name\": \"main\",\n",
        "      \"stage\": \"projection\",\n",
        "      \"parent\": null,\n",
        "      \"target\": \"cim\",\n",
        "      \"reason\": \"static-weight projection `main` is CiM-friendly and maps cleanly to tiles\",\n",
        "      \"shape\": [\n",
        "        2,\n",
        "        2\n",
        "      ],\n",
        "      \"tile_count\": 4\n",
        "    }\n",
        "  ],\n",
        "  \"attention_blocks\": []\n",
        "}\n"
    );

    assert_eq!(manifest_json(&package.manifest).unwrap(), expected);
}

#[test]
fn golden_memtorch_weight_payload_for_tiny_projection() {
    let normalized = tiny_program();
    let lowered = lower_program(&normalized, CompileConfig::square(1, 8)).unwrap();
    let package = build_package(&normalized, &lowered).unwrap();

    assert_eq!(package.weights, vec![129, 0, 127, 127]);
}
