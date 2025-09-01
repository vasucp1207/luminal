use std::f32::consts::{FRAC_PI_2, TAU};

use eframe::{
    egui,
    egui::{Color32, Pos2, Vec2},
};
use egglog::Term;
use itertools::Itertools;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    TermType,
    LoopLevel,
}

pub type DisplayNode = (String, egui::Color32, NodeShape, Option<usize>);
pub type DisplayGraph = StableGraph<DisplayNode, ()>;

pub fn display_graph(graph: &StableGraph<impl TermToString, impl EdgeToString, Directed, u32>) {
    display_multiple_graphs(&[graph]);
}

pub fn display_multiple_graphs<T: TermToString, E: EdgeToString>(
    graphs: &[&StableGraph<T, E, Directed, u32>],
) {
    // Build independent display graphs/views
    let views: Vec<View> = graphs
        .iter()
        .map(|g| View::new(to_display_graph(g, &[])))
        .collect();

    eframe::run_native(
        "Luminal Debugger",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([if graphs.len() == 1 { 1000.0 } else { 2000.0 }, 1200.0]),
            ..Default::default()
        },
        Box::new(move |_cc| Ok(Box::new(Debugger::new(views)))),
    )
    .unwrap();
}

fn to_display_graph<T: TermToString, E: EdgeToString>(
    graph: &StableGraph<T, E, Directed, u32>,
    mark_nodes: &[(NodeIndex, Color32)],
) -> DisplayGraph {
    // 1) copy nodes/edges
    let mut dg = DisplayGraph::new();
    let mut map = FxHashMap::default();
    for node in graph.node_indices() {
        let (r, c, b) = graph.node_weight(node).unwrap().term_to_string();
        map.insert(
            node,
            dg.add_node(
                if let Some((_, color)) = mark_nodes.iter().find(|(i, _)| *i == node) {
                    (r, *color, b, None)
                } else {
                    (r, c, b, None)
                },
            ),
        );
    }
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        dg.add_edge(map[&u], map[&v], ());
    }

    // 2) derive loop levels
    let mut seen = dg
        .externals(Direction::Incoming)
        .chain(dg.externals(Direction::Outgoing))
        .collect::<FxHashSet<_>>();

    let mut dfs = seen
        .iter()
        .flat_map(|n| dg.neighbors_directed(*n, Direction::Incoming))
        .collect_vec();

    while let Some(n) = dfs.pop() {
        if seen.contains(&n) {
            continue;
        }
        seen.insert(n);
        let curr_term = dg[n].0.clone();

        if let Some(outgoing_neighbor) = dg
            .neighbors_directed(n, Direction::Outgoing)
            .find(|x| seen.contains(x))
        {
            let (neighbor_weight, _, _, mut neighbor_levels) = dg[outgoing_neighbor].clone();
            if neighbor_weight.contains("LoopOut") {
                neighbor_levels = Some(neighbor_levels.unwrap_or_default() + 1);
            }
            if curr_term.contains("LoopIn") {
                neighbor_levels = neighbor_levels.map(|i| i - 1);
            }
            dg.node_weight_mut(n).unwrap().3 = neighbor_levels;
        } else if let Some(incoming_neighbor) = dg
            .neighbors_directed(n, Direction::Incoming)
            .find(|x| seen.contains(x))
        {
            let (neighbor_weight, _, _, mut neighbor_levels) = dg[incoming_neighbor].clone();
            if neighbor_weight.contains("LoopIn") {
                neighbor_levels = Some(neighbor_levels.unwrap_or_default() + 1);
            }
            if curr_term.contains("LoopOut") {
                neighbor_levels = neighbor_levels.map(|i| i - 1);
            }
            dg.node_weight_mut(n).unwrap().3 = neighbor_levels;
        }
        dfs.extend(dg.neighbors_undirected(n));
    }
    for (w, _, _, l) in dg.node_weights_mut() {
        if !w.contains("LoopIn") && !w.contains("LoopOut") {
            *l = None;
        }
    }

    dg
}

pub struct View {
    g: DisplayGraph,
    pos: Vec<Pos2>,
    dragging: Option<usize>,
    cam: Camera,
    need_fit: bool,
}

impl View {
    pub fn new(g: DisplayGraph) -> Self {
        Self {
            pos: layered_layout(&g, 140.0, 120.0, 100.0),
            g,
            dragging: None,
            cam: Camera::new(),
            need_fit: true,
        }
    }
}

pub struct Debugger {
    views: Vec<View>,
}

impl Debugger {
    pub fn new(views: Vec<View>) -> Self {
        Self { views }
    }
}

impl eframe::App for Debugger {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = Color32::BLACK;
        visuals.extreme_bg_color = Color32::BLACK;
        ctx.set_visuals(visuals);

        // Global 'D' toggles display mode for all views
        if ctx.input(|i| i.key_pressed(egui::Key::D)) {
            for v in &mut self.views {
                v.cam.toggle_mode();
            }
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("top_multi").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Pan: drag/bg scroll • Zoom: ⌘/Ctrl+wheel or pinch • Toggle mode: D");
                if ui.button("Fit all").clicked() {
                    for v in &mut self.views {
                        v.need_fit = true;
                    }
                }
                ui.separator();
                for (i, v) in self.views.iter().enumerate() {
                    ui.label(format!(
                        "View {i}: {:.1}x {:?}",
                        v.cam.zoom, v.cam.display_mode
                    ));
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let n = self.views.len().max(1);
            ui.columns(n, |cols| {
                for i in 0..n {
                    let ui_col = &mut cols[i];

                    // Stable painter per column
                    let viewport = ui_col.available_size();
                    let (resp, painter) =
                        ui_col.allocate_painter(viewport, egui::Sense::click_and_drag());
                    let origin = resp.rect.min;
                    let v = &mut self.views[i];

                    // Fit per view
                    if v.need_fit {
                        let (min_w, max_w) =
                            world_bounds(ui_col, &v.g, &v.pos, &egui::FontId::proportional(14.0));
                        if min_w.x.is_finite() {
                            v.cam.fit(origin, viewport, min_w, max_w, 40.0);
                        }
                        v.need_fit = false;
                    }

                    // Read input (global), then gate by hovered/active
                    let (mods, raw_scroll, pinch, pointer, drag_delta) = ui_col.input(|inp| {
                        (
                            inp.modifiers,
                            inp.raw_scroll_delta,
                            inp.zoom_delta_2d(),
                            inp.pointer.clone(),
                            inp.pointer.delta(),
                        )
                    });
                    let hovered = resp.hovered();
                    let active = hovered || v.dragging.is_some();

                    // Zoom: only when hovered/active and cursor inside this rect
                    if active {
                        let do_pinch = (pinch.y - 1.0).abs() > 1e-3 && hovered;
                        let do_wheel_zoom =
                            hovered && (mods.command || mods.ctrl) && raw_scroll.y != 0.0;

                        if do_pinch || do_wheel_zoom {
                            if let Some(cursor) = pointer.hover_pos() {
                                if resp.rect.contains(cursor) {
                                    let factor = if do_pinch {
                                        pinch.y
                                    } else {
                                        (raw_scroll.y * -0.001).exp()
                                    };
                                    v.cam.zoom_at(origin, cursor, factor);
                                    ui_col.ctx().request_repaint();
                                }
                            }
                        } else if hovered && raw_scroll != Vec2::ZERO {
                            // Trackpad/mouse scroll pans only when hovered
                            v.cam.pan += raw_scroll;
                        }
                    }

                    // Dragging: start only on this column; continue while pressed even if cursor leaves
                    if active {
                        if let Some(pos) = pointer.interact_pos() {
                            let pos_in_this = resp.rect.contains(pos);

                            if hovered && pointer.primary_pressed() && v.dragging.is_none() {
                                let pick_r2 = 16.0 * 16.0;
                                v.dragging = v
                                    .pos
                                    .iter()
                                    .enumerate()
                                    .map(|(j, &w)| (j, v.cam.w2s(origin, w).distance_sq(pos)))
                                    .filter(|&(_, d2)| d2 <= pick_r2)
                                    .min_by(|a, b| a.1.total_cmp(&b.1))
                                    .map(|(j, _)| j);
                            }

                            if pointer.primary_down() && (v.dragging.is_some() || pos_in_this) {
                                if let Some(j) = v.dragging {
                                    v.pos[j] = v.cam.s2w(origin, pos);
                                } else if hovered {
                                    v.cam.pan += drag_delta;
                                }
                            } else if pointer.primary_released() {
                                v.dragging = None;
                            }
                        }
                    }

                    // Draw
                    let z = v.cam.zoom;
                    let node_r = (14.0 * z).clamp(6.0, 40.0);
                    let label_dy = 22.0 * z;
                    let font_px = (14.0 * z).clamp(9.0, 48.0);
                    let font_id = egui::FontId::proportional(font_px);

                    for e in v.g.edge_indices() {
                        let (u, w) = v.g.edge_endpoints(e).unwrap();
                        let pu = v.cam.w2s(origin, v.pos[u.index()]);
                        let pw = v.cam.w2s(origin, v.pos[w.index()]);
                        draw_arrow(&painter, pu, pw, z, Color32::DARK_GRAY);
                    }
                    for nidx in v.g.node_indices() {
                        let p = v.cam.w2s(origin, v.pos[nidx.index()]);
                        let (label, _, _, _) = &v.g[nidx];
                        draw_node(&painter, p, node_r, &v.g[nidx], v.cam.display_mode);
                        painter.text(
                            p + Vec2::new(0.0, -label_dy),
                            egui::Align2::CENTER_CENTER,
                            label,
                            font_id.clone(),
                            Color32::WHITE,
                        );
                    }
                    painter.rect_stroke(resp.rect, 0.0, egui::Stroke::new(3.0, Color32::GREEN));
                }
            });
        });
    }
}
#[derive(Clone, Copy)]
pub struct Camera {
    zoom: f32, // scale
    pan: Vec2, // screen px offset
    display_mode: DisplayMode,
}

impl Camera {
    fn new() -> Self {
        Self {
            zoom: 1.0,
            pan: Vec2::ZERO,
            display_mode: DisplayMode::TermType,
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
    fn fit(
        &mut self,
        _origin: Pos2,
        viewport: Vec2,
        world_min: Pos2,
        world_max: Pos2,
        margin: f32,
    ) {
        // world extents
        let size_w = (world_max - world_min).max(Vec2::splat(1.0));
        // usable viewport *in local coords* (no origin baked in)
        let usable = (viewport - Vec2::splat(2.0 * margin)).max(Vec2::splat(1.0));

        self.zoom = (usable.x / size_w.x)
            .min(usable.y / size_w.y)
            .clamp(0.1, 10.0);

        // Map world_min -> margin, then center any extra space. Stay in local coords.
        let mapped_min_local = Vec2::splat(margin);
        let mapped_size = size_w * self.zoom;
        let extra = (usable - mapped_size).max(Vec2::ZERO) * 0.5;

        // IMPORTANT: no origin here. w2s will add origin later.
        self.pan = mapped_min_local + extra - world_min.to_vec2() * self.zoom;
    }
    #[inline]
    fn toggle_mode(&mut self) {
        self.display_mode = match self.display_mode {
            DisplayMode::TermType => DisplayMode::LoopLevel,
            DisplayMode::LoopLevel => DisplayMode::TermType,
        };
    }
}

// Small helper to draw nodes
fn draw_node(
    p: &egui::Painter,
    c: Pos2,
    r: f32,
    (_, color, shape, loop_level): &DisplayNode,
    display_mode: DisplayMode,
) {
    let color = if display_mode == DisplayMode::LoopLevel
        && let Some(loop_level) = loop_level
    {
        rainbow_color(*loop_level)
    } else {
        *color
    };
    match shape {
        NodeShape::Circle => {
            p.circle_filled(c, r, color);
        }
        NodeShape::Triangle => {
            let pts: Vec<_> = (0..3)
                .map(|i| {
                    let ang = TAU * (i as f32) / 3.0 - FRAC_PI_2; // up
                    Pos2::new(c.x + r * ang.cos(), c.y + r * ang.sin())
                })
                .collect();
            p.add(egui::epaint::Shape::convex_polygon(
                pts.clone(),
                color,
                egui::Stroke::NONE,
            ));
        }
        NodeShape::InvertedTriangle => {
            let pts: Vec<_> = (0..3)
                .map(|i| {
                    let ang = TAU * (i as f32) / 3.0 + FRAC_PI_2; // down
                    Pos2::new(c.x + r * ang.cos(), c.y + r * ang.sin())
                })
                .collect();
            p.add(egui::epaint::Shape::convex_polygon(
                pts.clone(),
                color,
                egui::Stroke::NONE,
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
                pts.clone(),
                color,
                egui::Stroke::NONE,
            ));
        }
    }
}

fn rainbow_color(i: usize) -> egui::Color32 {
    // Map 0..=10 across 0..=360 degrees
    let hue = (9.0 - i as f32) / 10.0 * 360.0;
    hsv_to_rgb(hue, 1.0, 1.0)
}

// Simple HSV → Color32
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> egui::Color32 {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h as i32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    egui::Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
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
            GraphTerm::LoopIn {
                range,
                stride,
                marker,
            } => (
                format!("LoopIn ({range}; {stride}; ({marker}))"),
                Color32::LIGHT_BLUE,
                NodeShape::InvertedTriangle,
            ),
            GraphTerm::LoopOut {
                range,
                stride,
                marker,
            } => (
                format!("LoopOut ({range}; {stride}; ({marker}))"),
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
