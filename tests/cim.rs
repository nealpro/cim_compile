use std::str::FromStr;

use cim_compile::cim::{
    MatrixShape, Program, ProjectionKind, TileCoord, TileDispatch, TileSize, expected_weight_offset,
};

fn tiny_dispatch(order: u32, row: u32, col: u32) -> TileDispatch {
    TileDispatch {
        projection: ProjectionKind::WQ,
        tile: TileCoord::new(row, col),
        matrix_shape: MatrixShape::new(2, 2),
        tile_size: TileSize::new(1, 1),
        weight_offset: expected_weight_offset(order, TileSize::new(1, 1)),
        order,
    }
}

fn tiny_program() -> Program {
    Program::new(
        "dialect_test",
        "dispatch",
        vec![
            tiny_dispatch(0, 0, 0),
            tiny_dispatch(1, 0, 1),
            tiny_dispatch(2, 1, 0),
            tiny_dispatch(3, 1, 1),
        ],
    )
}

const EXPECTED_TINY_TEXT: &str = concat!(
    "cim.module @dialect_test {\n",
    "  cim.func @dispatch {\n",
    "    cim.tile.dispatch { projection = \"wq\", tile = [0, 0], matrix_shape = [2, 2], tile_size = [1, 1], weight_offset = 0, order = 0 }\n",
    "    cim.tile.dispatch { projection = \"wq\", tile = [0, 1], matrix_shape = [2, 2], tile_size = [1, 1], weight_offset = 4, order = 1 }\n",
    "    cim.tile.dispatch { projection = \"wq\", tile = [1, 0], matrix_shape = [2, 2], tile_size = [1, 1], weight_offset = 8, order = 2 }\n",
    "    cim.tile.dispatch { projection = \"wq\", tile = [1, 1], matrix_shape = [2, 2], tile_size = [1, 1], weight_offset = 12, order = 3 }\n",
    "  }\n",
    "}\n",
);

#[test]
fn cim_printer_emits_canonical_text() {
    let program = tiny_program();

    assert_eq!(program.to_text(), EXPECTED_TINY_TEXT);
}

#[test]
fn cim_round_trips_through_text() {
    let parsed = Program::from_str(EXPECTED_TINY_TEXT).expect("expected tiny cim text to parse");

    assert_eq!(parsed, tiny_program());
}

#[test]
fn cim_rejects_empty_program() {
    let program = Program::new("dialect_test", "dispatch", vec![]);

    let err = program.verify().unwrap_err();
    assert!(err.contains("at least one cim.tile.dispatch"), "{err}");
}

#[test]
fn cim_rejects_bad_weight_offset() {
    let mut program = tiny_program();
    program.entry.dispatches[2].weight_offset += 1;

    let err = program.verify().unwrap_err();
    assert!(err.contains("weight_offset"), "{err}");
}

#[test]
fn cim_rejects_non_monotonic_order() {
    let mut program = tiny_program();
    program.entry.dispatches[0].order = 3;
    program.entry.dispatches[0].weight_offset = expected_weight_offset(3, TileSize::new(1, 1));
    program.entry.dispatches[3].order = 0;
    program.entry.dispatches[3].weight_offset = expected_weight_offset(0, TileSize::new(1, 1));

    let err = program.verify().unwrap_err();
    assert!(err.contains("inconsistent dispatch schedule"), "{err}");
}

#[test]
fn cim_rejects_invalid_tile_size() {
    let mut program = tiny_program();
    for dispatch in &mut program.entry.dispatches {
        dispatch.tile_size = TileSize::new(0, 1);
    }

    let err = program.verify().unwrap_err();
    assert!(
        err.contains("tile_size dimensions must be greater than zero"),
        "{err}"
    );
}

#[test]
fn cim_rejects_illegal_projection_name() {
    let text = EXPECTED_TINY_TEXT.replace("\"wq\"", "\"bad name\"");

    let err = Program::from_str(&text).unwrap_err();
    assert!(err.contains("invalid projection name"), "{err}");
}

#[test]
fn cim_expected_weight_offsets_match_layout() {
    let tile_size = TileSize::new(1, 1);

    assert_eq!(expected_weight_offset(0, tile_size), 0);
    assert_eq!(expected_weight_offset(3, tile_size), 12);
}
