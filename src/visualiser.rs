use crate::{Individual, Population, RealSolution, Tracker};
use std::{ops::Range, path::Path};

/// A trait for individuals that can be converted to a real-valued point in `N`-dimensional space.
pub trait AsRealPoint<const N: usize> {
    fn as_real_point(&self) -> [f64; N];
}
impl<const N: usize> AsRealPoint<N> for RealSolution<N> {
    fn as_real_point(&self) -> [f64; N] {
        self.inner.map(|x| x.into_inner())
    }
}

/// A tracker that observes the history of all the candidate solutions attempted in a simulation,
/// and then allows plotting the evolution of the simulation afterward. This supports either 2D or
/// 3D plotting, and will work with any individual whose fitness value can be represented as an
/// `f64` (strictly, [`Individual::Fitness`] must implement [`AsRef<f64>`]) and whose inner value
/// implements [`AsRealPoint<N>`], where `N` is the dimensionality of the solution space. The
/// resulting plot will be in `N + 1`-dimensional space (an extra dimension for the fitness).
///
/// For clarity, a one-dimensional optimisation will produce a two-dimensional plot, and a
/// two-dimensional optimisation will produce a three-dimensional plot. In both cases, the fitness
/// will be represented by the y-axis. The generic parameter `N` is for the dimensionality of the
/// optimisation, not the final plot.
pub struct VisualTracker<I: Individual, const N: usize> {
    /// All the points that have ever been proposed as candidate solutions, ordered from the first
    /// generation to the latest. Within each generation, the points are ordered from worst to best
    /// fitness.
    points: Vec<Vec<([f64; N], f64)>>,
    /// A function that maps the individual's fitness type to an `f64` for plotting as the
    /// y-component of each individual vector. This might reciprocate for minimisation problems.
    fitness_map: Box<dyn Fn(I::Fitness) -> f64>,
    /// A function representing the curve/surface being optimised over, which will be plotted.
    objective_fn: fn([f64; N]) -> f64,
}
impl<I: Individual, const N: usize> VisualTracker<I, N> {
    /// Creates a new visual tracker for the given objective function, which will be plotted in one
    /// higher than the dimensionality of `N` (the extra dimension being for the fitness returned
    /// by the objective function). This takes a function that maps the fitness values of
    /// individuals to `f64`s for plotting (e.g. minimisation problems will need to use this to
    /// convert their values back to ones suitable for plotting).
    ///
    /// The objective function provided should be a raw function pointer representing the
    /// *mathematical* function being optimised, and this will be plotted as a curve/surface.
    pub fn new_custom(
        objective_fn: fn([f64; N]) -> f64,
        fitness_map: impl Fn(I::Fitness) -> f64 + 'static,
    ) -> Self {
        Self {
            points: Vec::new(),
            fitness_map: Box::new(fitness_map),
            objective_fn,
        }
    }

    /// Directly gets all the points recorded throughout the history of the simulation. This will
    /// be a vector of vectors, where each inner vector represents a generation, and each element
    /// of that vector is a tuple of the candidate solution and its fitness (all as `f64`s).
    pub fn points(&self) -> &Vec<Vec<([f64; N], f64)>> {
        &self.points
    }

    /// Gets a mutable reference to the points recorded throughout the history of the simulation.
    /// See [`Self::points`] for details.
    pub fn points_mut(&mut self) -> &mut Vec<Vec<([f64; N], f64)>> {
        &mut self.points
    }

    /// Consumes the visual tracker and returns the points recorded throughout the history of the
    /// simulation. See [`Self::points`] for details.
    pub fn into_points(self) -> Vec<Vec<([f64; N], f64)>> {
        self.points
    }
}
impl<I: Individual, const N: usize> VisualTracker<I, N>
where
    I::Fitness: AsRef<f64>,
{
    /// Creates a new visual tracker for maximising the given objective function (the default).
    /// This should be used when the fitness values of individuals are directly representative of
    /// what should be plotted.
    ///
    /// The objective function provided should be a raw function pointer representing the
    /// *mathematical* function being optimised, and this will be plotted as a curve/surface.
    pub fn new_max(objective_fn: fn([f64; N]) -> f64) -> Self {
        Self::new_custom(objective_fn, |f| *f.as_ref())
    }

    /// Creates a new visual tracker for minimising the given objective function. This should be
    /// used when the fitness values of individuals have been modified as `1.0 / x` to invert the
    /// problem.
    ///
    /// The objective function provided should be a raw function pointer representing the
    /// *mathematical* function being optimised, and this will be plotted as a curve/surface.
    pub fn new_min(objective_fn: fn([f64; N]) -> f64) -> Self {
        Self::new_custom(objective_fn, |f| 1.0 / *f.as_ref())
    }
}

/// A trait for visualising the evolution of a simulation, producing a GIF of its evolution in
/// `N + 1`-dimensional space. This is a marker trait that can be used to ensure a given `N` has an
/// implementation for visualising the simulation.
pub trait Visualise<const N: usize> {
    fn visualise(
        &self,
        filename: &Path,
        title: &str,
        input_ranges: [Range<f64>; N],
        output_range: Range<f64>,
        axis_step: f64,
        frame_delay: u32,
    );
}

impl<I: Individual> Visualise<2> for VisualTracker<I, 2> {
    /// Visualises the evolution of the simulation as a GIF, saving it to the given filename. The
    /// title will be displayed at the top of the plot, and the axis ranges will apply for each
    /// axis in order (i.e. x, y, z), each with the given step. The frame delay controls how many
    /// milliseconds each generation is shown for (100-250 tend to be good values).
    fn visualise(
        &self,
        filename: &Path,
        title: &str,
        axis_ranges: [Range<f64>; 2],
        output_range: Range<f64>,
        axis_step: f64,
        frame_delay: u32,
    ) {
        use plotters::prelude::*;

        let area = BitMapBackend::gif(filename, (1024, 1024), frame_delay)
            .unwrap()
            .into_drawing_area();

        // Plot each generation
        for points in &self.points {
            area.fill(&WHITE).unwrap();

            let mut chart = ChartBuilder::on(&area)
                .caption(title, ("sans-serif", 20))
                .build_cartesian_3d(
                    axis_ranges[0].clone().step(axis_step),
                    output_range.clone().step(axis_step),
                    axis_ranges[1].clone().step(axis_step),
                )
                .unwrap();

            chart.with_projection(|mut pb| {
                pb.yaw = 0.5;
                pb.scale = 0.85;
                pb.into_matrix()
            });

            chart
                .configure_axes()
                .light_grid_style(BLACK.mix(0.15))
                .max_light_lines(3)
                .draw()
                .unwrap();

            let render_step = axis_step * 10.0;

            chart
                .draw_series(
                    SurfaceSeries::xoz(
                        // We want to plot the function at every step, which means we need to
                        // convert something like -5.0..5.0 with a step of 0.1 to -50..50 and then
                        // evaluate on every one of those points, times 0.1
                        (((axis_ranges[0].start / render_step) as i64)
                            ..((axis_ranges[0].end / render_step) as i64))
                            .map(|f| f as f64 * render_step),
                        (((axis_ranges[1].start / render_step) as i64)
                            ..((axis_ranges[1].end / render_step) as i64))
                            .map(|f| f as f64 * render_step),
                        |x, z| (self.objective_fn)([x, z]),
                    )
                    .style(BLUE.mix(0.2).filled()),
                )
                .unwrap()
                .label("Surface")
                .legend(|(x, y)| {
                    Rectangle::new([(x + 5, y - 5), (x + 15, y + 5)], BLUE.mix(0.5).filled())
                });

            chart
                .draw_series(PointSeries::<(f64, f64, f64), _, Circle<_, _>, _>::new(
                    points.iter().map(|(xz, y)| (xz[0], *y, xz[1])),
                    1,
                    ShapeStyle::from(BLACK).filled(),
                ))
                .unwrap()
                .label("Candidate solutions")
                .legend(|(x, y)| Circle::new((x + 5, y + 5), 3, BLACK.filled()));

            chart
                .configure_series_labels()
                .border_style(BLACK)
                .draw()
                .unwrap();

            area.present().unwrap();
        }

        area.present().unwrap();
    }
}
impl<I: Individual> Visualise<1> for VisualTracker<I, 1> {
    /// Visualises the evolution of the simulation as a GIF, saving it to the given filename. The
    /// title will be displayed at the top of the plot, and the axis ranges will apply for each
    /// axis in order (i.e. x, y, z), each with the given step. The frame delay controls how many
    /// milliseconds each generation is shown for (100-250 tend to be good values).
    fn visualise(
        &self,
        filename: &Path,
        title: &str,
        axis_ranges: [Range<f64>; 1],
        output_range: Range<f64>,
        axis_step: f64,
        frame_delay: u32,
    ) {
        use plotters::prelude::*;

        let area = BitMapBackend::gif(filename, (1024, 1024), frame_delay)
            .unwrap()
            .into_drawing_area();

        // Plot each generation
        for points in &self.points {
            area.fill(&WHITE).unwrap();

            let mut chart = ChartBuilder::on(&area)
                .caption(title, ("sans-serif", 20))
                .build_cartesian_2d(
                    axis_ranges[0].clone().step(axis_step),
                    output_range.clone().step(axis_step),
                )
                .unwrap();

            let render_step = axis_step * 10.0;

            chart
                .draw_series(LineSeries::new(
                    // We want to plot the function at every step, which means we need to
                    // convert something like -5.0..5.0 with a step of 0.1 to -50..50 and then
                    // evaluate on every one of those points, times 0.1
                    (((axis_ranges[0].start / render_step) as i64)
                        ..((axis_ranges[0].end / render_step) as i64))
                        .map(|f| f as f64 * render_step)
                        .map(|x| (x, (self.objective_fn)([x]))),
                    BLUE.mix(0.2).filled(),
                ))
                .unwrap()
                .label("Surface")
                .legend(|(x, y)| {
                    Rectangle::new([(x + 5, y - 5), (x + 15, y + 5)], BLUE.mix(0.5).filled())
                });

            chart
                .draw_series(PointSeries::<(f64, f64), _, Circle<_, _>, _>::new(
                    points.iter().map(|(x, y)| (x[0], *y)),
                    1,
                    ShapeStyle::from(BLACK).filled(),
                ))
                .unwrap()
                .label("Candidate solutions")
                .legend(|(x, y)| Circle::new((x + 5, y + 5), 3, BLACK.filled()));

            chart
                .configure_series_labels()
                .border_style(BLACK)
                .draw()
                .unwrap();

            area.present().unwrap();
        }

        area.present().unwrap();
    }
}

// Any individual that can be converted to a 2D real vector can be visualised in 3D, with its
// fitness value on the y-axis
impl<I: Individual + AsRealPoint<2>> Tracker<I> for VisualTracker<I, 2> {
    fn process_generation(&mut self, population: &Population<I>, _previous_generation: &Vec<I>) {
        let new_points = population
            .individuals()
            .map(|i| {
                let xz = i.into_inner().as_real_point();
                let y = (self.fitness_map)(i.fitness());

                ([xz[0], xz[1]], y)
            })
            .collect();
        self.points.push(new_points);
    }
}

// 1D optimisations -> 2D plots
impl<I: Individual + AsRealPoint<1>> Tracker<I> for VisualTracker<I, 1> {
    fn process_generation(&mut self, population: &Population<I>, _previous_generation: &Vec<I>) {
        let new_points = population
            .individuals()
            .map(|i| {
                let x = i.into_inner().as_real_point();
                let y = (self.fitness_map)(i.fitness());

                ([x[0]], y)
            })
            .collect();
        self.points.push(new_points);
    }
}
