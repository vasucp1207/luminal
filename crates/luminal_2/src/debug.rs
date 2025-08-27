use std::f32::consts::{FRAC_PI_2, TAU};

use eframe::{
    egui,
    egui::{Color32, Pos2, Vec2},
};
use egglog::Term;
use luminal::{
    prelude::petgraph::{Directed, Direction, graph::NodeIndex, prelude::StableGraph},
    shape::Expression,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{GraphTerm, Kernel};

#[derive(Debug, Clone, Copy)]
pub enum NodeShape {
    Circle,
    Triangle,
    InvertedTriangle,
    Square,
}

type NodeData = (String, egui::Color32, NodeShape);
type EdgeData = ();
type DisplayGraph = StableGraph<NodeData, EdgeData>;

pub fn display_graph(
    graph: &StableGraph<impl TermToString, impl EdgeToString, Directed, u32>,
    mark_nodes: &[(NodeIndex, Color32)],
) {
    // Convert graph into a displayable graph
    let mut display_graph = DisplayGraph::new();
    let mut map = FxHashMap::default();
    for node in graph.node_indices() {
        map.insert(
            node,
            display_graph.add_node(
                if let Some((_, color)) = mark_nodes.iter().find(|(i, _)| *i == node) {
                    (
                        graph.node_weight(node).unwrap().term_to_string().0,
                        *color,
                        graph.node_weight(node).unwrap().term_to_string().2,
                    )
                } else {
                    graph.node_weight(node).unwrap().term_to_string()
                },
            ),
        );
    }
    for edge in graph.edge_indices() {
        let (start, end) = graph.edge_endpoints(edge).unwrap();
        display_graph.add_edge(map[&start], map[&end], ());
    }

    eframe::run_native(
        "Luminal Debugger",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1000 as f32, 1200 as f32]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(Debugger::new(display_graph)))),
    )
    .unwrap();
}

struct Debugger {
    g: DisplayGraph,
    pos: Vec<Pos2>,
    dragging: Option<usize>,
    cam: Camera,
    need_fit: bool,
}

#[derive(Clone, Copy)]
struct Camera {
    zoom: f32, // scale
    pan: Vec2, // screen px offset
}

impl Camera {
    fn new() -> Self {
        Self {
            zoom: 1.0,
            pan: Vec2::ZERO,
        }
    }

    #[inline]
    fn w2s(&self, origin: Pos2, w: Pos2) -> Pos2 {
        origin + (self.pan + w.to_vec2() * self.zoom)
    }
    #[inline]
    fn s2w(&self, origin: Pos2, s: Pos2) -> Pos2 {
        (((s - origin) - self.pan) / self.zoom).to_pos2()
    }
    fn zoom_at(&mut self, origin: Pos2, cursor: Pos2, factor: f32) {
        let world_before = self.s2w(origin, cursor);
        let new_zoom = (self.zoom * factor).clamp(0.1, 10.0);
        self.pan = (cursor - origin) - world_before.to_vec2() * new_zoom;
        self.zoom = new_zoom;
    }
    fn fit(&mut self, origin: Pos2, viewport: Vec2, world_min: Pos2, world_max: Pos2, margin: f32) {
        let size_w = (world_max - world_min).max(Vec2::splat(1.0));
        let usable = (viewport - Vec2::splat(2.0 * margin)).max(Vec2::splat(1.0));
        self.zoom = (usable.x / size_w.x)
            .min(usable.y / size_w.y)
            .clamp(0.1, 10.0);
        let mapped_min = origin + Vec2::splat(margin);
        let mapped_size = size_w * self.zoom;
        let extra = (usable - mapped_size).max(Vec2::ZERO) * 0.5;
        self.pan = ((mapped_min + extra) - world_min.to_vec2() * self.zoom).to_vec2();
    }
}

impl Debugger {
    fn new(g: DisplayGraph) -> Self {
        Self {
            pos: layered_layout(&g, 140.0, 120.0, 100.0),
            g,
            dragging: None,
            cam: Camera::new(),
            need_fit: true,
        }
    }
}

impl eframe::App for Debugger {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Drag nodes. Drag background / two-finger scroll to pan. ⌘/Ctrl+wheel or pinch to zoom.");
                if ui.button("Fit").clicked() { self.need_fit = true; }
                ui.label(format!("Zoom: {:.1}x", self.cam.zoom));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Stable viewport-sized painter
            let viewport = ui.available_size();
            let (resp, painter) = ui.allocate_painter(viewport, egui::Sense::click_and_drag());
            let origin = resp.rect.min;

            // Fit once (or after button)
            if self.need_fit {
                let (min_w, max_w) =
                    world_bounds(ui, &self.g, &self.pos, &egui::FontId::proportional(14.0));
                if min_w.x.is_finite() {
                    self.cam.fit(origin, viewport, min_w, max_w, 40.0);
                }
                self.need_fit = false;
            }

            // --- Input handling ---
            let (mods, raw_scroll, pinch, pointer, drag_delta) = ui.input(|i| {
                (
                    i.modifiers,
                    i.raw_scroll_delta,
                    i.zoom_delta_2d(),
                    i.pointer.clone(),
                    i.pointer.delta(),
                )
            });

            // Zoom (pinch or Cmd/Ctrl + wheel), anchored at cursor (fallback to center)
            let do_pinch = (pinch.y - 1.0).abs() > 1e-3;
            let do_wheel_zoom = (mods.command || mods.ctrl) && raw_scroll.y != 0.0;
            if do_pinch || do_wheel_zoom {
                let cursor = pointer.hover_pos().unwrap_or(resp.rect.center());
                let factor = if do_pinch {
                    pinch.y
                } else {
                    (raw_scroll.y * -0.001).exp()
                };
                self.cam.zoom_at(origin, cursor, factor);
                ui.ctx().request_repaint();
            } else if raw_scroll != Vec2::ZERO {
                // Trackpad scroll pans when not zooming
                self.cam.pan += raw_scroll;
            }

            // Node picking/dragging or background pan drag
            if let Some(pos) = pointer.interact_pos() {
                if pointer.primary_pressed() && self.dragging.is_none() {
                    let pick_r2 = 16.0 * 16.0;
                    self.dragging = self
                        .pos
                        .iter()
                        .enumerate()
                        .map(|(i, &w)| (i, self.cam.w2s(origin, w).distance_sq(pos)))
                        .filter(|&(_, d2)| d2 <= pick_r2)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(i, _)| i);
                }
                if pointer.primary_down() {
                    if let Some(i) = self.dragging {
                        self.pos[i] = self.cam.s2w(origin, pos);
                    } else {
                        self.cam.pan += drag_delta;
                    }
                } else if pointer.primary_released() {
                    self.dragging = None;
                }
            }

            // --- Drawing ---
            let z = self.cam.zoom;
            let node_r = (14.0 * z).clamp(6.0, 40.0);
            let label_dy = 22.0 * z;
            let font_px = (14.0 * z).clamp(9.0, 48.0);
            let node_stroke = (1.0 * z).clamp(0.5, 3.0);

            // edges
            for e in self.g.edge_indices() {
                let (u, v) = self.g.edge_endpoints(e).unwrap();
                let pu = self.cam.w2s(origin, self.pos[u.index()]);
                let pv = self.cam.w2s(origin, self.pos[v.index()]);
                draw_arrow(&painter, pu, pv, z, Color32::DARK_GRAY);
            }

            // nodes + labels
            let font_id = egui::FontId::proportional(font_px);
            for n in self.g.node_indices() {
                let p = self.cam.w2s(origin, self.pos[n.index()]);
                let (label, color, shape) = &self.g[n];
                draw_node(&painter, p, *color, *shape, node_r, node_stroke);
                painter.text(
                    p + Vec2::new(0.0, -label_dy),
                    egui::Align2::CENTER_CENTER,
                    label,
                    font_id.clone(),
                    Color32::WHITE,
                );
            }
        });
    }
}

// Small helper to draw nodes
fn draw_node(p: &egui::Painter, c: Pos2, color: Color32, shape: NodeShape, r: f32, stroke_w: f32) {
    match shape {
        NodeShape::Circle => {
            p.circle_filled(c, r, color);
            p.circle_stroke(c, r, egui::Stroke::new(stroke_w, Color32::BLACK));
        }
        NodeShape::Triangle => {
            let mut pts = [Pos2::ZERO; 3];
            for i in 0..3 {
                let ang = TAU * (i as f32) / 3.0 - FRAC_PI_2; // up
                pts[i] = Pos2::new(c.x + r * ang.cos(), c.y + r * ang.sin());
            }
            p.add(egui::epaint::Shape::convex_polygon(
                pts.to_vec(),
                color,
                egui::Stroke::new(1.0, color),
            ));
        }
        NodeShape::InvertedTriangle => {
            let mut pts = [Pos2::ZERO; 3];
            for i in 0..3 {
                let ang = TAU * (i as f32) / 3.0 + FRAC_PI_2; // down
                pts[i] = Pos2::new(c.x + r * ang.cos(), c.y + r * ang.sin());
            }
            p.add(egui::epaint::Shape::convex_polygon(
                pts.to_vec(),
                color,
                egui::Stroke::new(1.0, color),
            ));
        }
        NodeShape::Square => {
            let pts = vec![
                Pos2::new(c.x - r, c.y - r),
                Pos2::new(c.x + r, c.y - r),
                Pos2::new(c.x + r, c.y + r),
                Pos2::new(c.x - r, c.y + r),
            ];
            p.add(egui::epaint::Shape::convex_polygon(
                pts,
                color,
                egui::Stroke::new(1.0, Color32::BLACK),
            ));
        }
    }
}

fn world_bounds(
    ui: &mut egui::Ui,
    g: &DisplayGraph,
    pos: &[Pos2],
    font_id: &egui::FontId,
) -> (Pos2, Pos2) {
    const R: f32 = 14.0;
    const DY: f32 = 22.0;
    let mut min = Pos2::new(f32::INFINITY, f32::INFINITY);
    let mut max = Pos2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    ui.fonts(|f| {
        for n in g.node_indices() {
            let p = pos[n.index()];
            let galley = f.layout_no_wrap(g[n].0.clone(), font_id.clone(), Color32::WHITE);
            let hw = galley.size().x * 0.5;
            let hh = galley.size().y * 0.5;
            let label_c = Pos2::new(p.x, p.y - DY);
            let lmin = Pos2::new(label_c.x - hw, label_c.y - hh);
            let lmax = Pos2::new(label_c.x + hw, label_c.y + hh);
            let nmin = Pos2::new(p.x - R, p.y - R);
            let nmax = Pos2::new(p.x + R, p.y + R);
            min.x = min.x.min(lmin.x.min(nmin.x));
            min.y = min.y.min(lmin.y.min(nmin.y));
            max.x = max.x.max(lmax.x.max(nmax.x));
            max.y = max.y.max(lmax.y.max(nmax.y));
        }
    });
    if !min.x.is_finite() {
        (Pos2::ZERO, Pos2::new(400.0, 300.0))
    } else {
        (min, max)
    }
}

/// Simple layered DAG layout
fn layered_layout(g: &DisplayGraph, dx: f32, layer_dy: f32, layer_cx: f32) -> Vec<Pos2> {
    use std::collections::VecDeque;
    let n = g.node_count();
    let mut indeg = vec![0usize; n];
    for ni in g.node_indices() {
        indeg[ni.index()] = g.neighbors_directed(ni, Direction::Incoming).count();
    }
    let mut q = VecDeque::new();
    for ni in g.node_indices() {
        if indeg[ni.index()] == 0 {
            q.push_back(ni);
        }
    }
    let mut order = Vec::with_capacity(n);
    while let Some(u) = q.pop_front() {
        order.push(u);
        for v in g.neighbors_directed(u, Direction::Outgoing) {
            let i = v.index();
            indeg[i] -= 1;
            if indeg[i] == 0 {
                q.push_back(v);
            }
        }
    }
    let mut depth = vec![0usize; n];
    for &u in &order {
        let du = depth[u.index()];
        for v in g.neighbors_directed(u, Direction::Outgoing) {
            depth[v.index()] = depth[v.index()].max(du + 1);
        }
    }
    let max_d = depth.iter().copied().max().unwrap_or(0);
    let mut by_layer: Vec<Vec<NodeIndex>> = vec![vec![]; max_d + 1];
    for ni in g.node_indices() {
        by_layer[depth[ni.index()]].push(ni);
    }
    let mut pos = vec![Pos2::ZERO; n];
    for (layer, nodes) in by_layer.into_iter().enumerate() {
        let w = (nodes.len().saturating_sub(1)) as f32 * dx;
        for (k, ni) in nodes.into_iter().enumerate() {
            let x = (k as f32) * dx - w * 0.5 + layer_cx;
            let y = layer as f32 * layer_dy + 80.0;
            pos[ni.index()] = Pos2::new(x, y);
        }
    }
    pos
}

// replace your draw_arrow with a zoom-aware version
fn draw_arrow(p: &egui::Painter, from: Pos2, to: Pos2, zoom: f32, color: Color32) {
    let w = (2.0 * zoom).clamp(1.0, 4.0);
    let stroke = egui::Stroke::new(w, color);
    p.line_segment([from, to], stroke);
}

pub trait TermToString {
    fn term_to_string(&self) -> (String, Color32, NodeShape);
}

pub trait EdgeToString {
    fn edge_to_string(&self) -> String;
}

impl EdgeToString for usize {
    fn edge_to_string(&self) -> String {
        self.to_string()
    }
}

impl EdgeToString for () {
    fn edge_to_string(&self) -> String {
        "".to_string()
    }
}

impl EdgeToString for (usize, usize) {
    fn edge_to_string(&self) -> String {
        format!("{}, {}", self.0, self.1)
    }
}

impl TermToString for Term {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            match self {
                Term::App(a, _) => a.to_string(),
                Term::Lit(l) => l.to_string(),
                Term::Var(v) => v.to_string(),
            },
            Color32::GREEN,
            NodeShape::Circle,
        )
    }
}

impl TermToString for usize {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (self.to_string(), Color32::GREEN, NodeShape::Circle)
    }
}

impl TermToString for String {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (self.clone(), Color32::GREEN, NodeShape::Circle)
    }
}

impl TermToString for (Term, usize) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        let s = match &self.0 {
            Term::App(a, _) => a.to_string(),
            Term::Lit(l) => l.to_string(),
            Term::Var(v) => v.to_string(),
        };
        (
            format!("{s}[{}]", self.1),
            Color32::GREEN,
            NodeShape::Circle,
        )
    }
}

impl TermToString for Kernel {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            if self.code.starts_with("Inputs") {
                "Inputs".to_string()
            } else if self.code.starts_with("Outputs") {
                "Outputs".to_string()
            } else {
                format!(
                    "Kernel {:?} {:?} -> {:?}",
                    self.grid, self.threadblock, self.outputs
                )
            },
            Color32::GOLD,
            NodeShape::Square,
        )
    }
}

impl TermToString for GraphTerm {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        match self {
            GraphTerm::Add => ("Add".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::Mul => ("Mul".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::Max => ("Max".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::Exp2 => ("Exp2".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::Log2 => ("Log2".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::Sin => ("Sin".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::Recip => ("Recip".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::Neg => ("Neg".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::Sqrt => ("Sqrt".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::Mod => ("Mod".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::LessThan => ("LessThan".to_string(), Color32::GREEN, NodeShape::Circle),
            GraphTerm::TCMatmul {
                a_k_stride,
                b_k_stride,
                a_inner_stride,
                b_inner_stride,
                c_inner_stride,
                k_outer_loops,
            } => (
                format!(
                    "TCMatmul ({a_k_stride}, {b_k_stride}, {a_inner_stride}, {b_inner_stride}, {c_inner_stride}, {k_outer_loops})"
                ),
                Color32::RED,
                NodeShape::Circle,
            ),
            GraphTerm::LoopIn { range, stride, .. } => (
                format!("LoopIn ({range}; {stride})"),
                Color32::LIGHT_BLUE,
                NodeShape::InvertedTriangle,
            ),
            GraphTerm::LoopOut { range, stride, .. } => (
                format!("LoopOut ({range}; {stride})"),
                Color32::BLUE,
                NodeShape::Triangle,
            ),
            GraphTerm::GMEM { label } => {
                (format!("GMEM ({label})"), Color32::GOLD, NodeShape::Square)
            }
            GraphTerm::SMEM => ("SMEM".to_string(), Color32::YELLOW, NodeShape::Square),
            GraphTerm::Custom(_) => ("CustomKernel".to_string(), Color32::GRAY, NodeShape::Circle),
            GraphTerm::Diff(d) => (format!("Diff({d})"), Color32::GRAY, NodeShape::Circle),
            GraphTerm::SMEMLoad => ("SMEMLoad".to_string(), Color32::YELLOW, NodeShape::Circle),
            GraphTerm::SMEMRead => ("SMEMRead".to_string(), Color32::YELLOW, NodeShape::Circle),
            GraphTerm::Break => ("Break".to_string(), Color32::WHITE, NodeShape::Circle),
        }
    }
}

impl TermToString for (GraphTerm, usize) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            format!("{} [{}]", self.0.term_to_string().0, self.1),
            self.0.term_to_string().1,
            NodeShape::Circle,
        )
    }
}

impl TermToString for (GraphTerm, Vec<Expression>, Vec<usize>) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            format!(
                "{} {:?} {{{:?}}}",
                self.0.term_to_string().0,
                self.1,
                self.2
            ),
            self.0.term_to_string().1,
            NodeShape::Circle,
        )
    }
}

impl TermToString for (GraphTerm, Vec<Expression>, FxHashSet<usize>) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            format!(
                "{} {:?} {{{:?}}}",
                self.0.term_to_string().0,
                self.1,
                self.2
            ),
            self.0.term_to_string().1,
            NodeShape::Circle,
        )
    }
}

impl TermToString for (GraphTerm, Vec<String>, Vec<usize>) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            format!(
                "{} [{}] {{{:?}}}",
                self.0.term_to_string().0,
                self.1.len(),
                self.2
            ),
            self.0.term_to_string().1,
            NodeShape::Circle,
        )
    }
}
