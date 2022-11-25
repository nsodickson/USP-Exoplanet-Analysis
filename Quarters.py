from Pipeline import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

def produceQuarterPeriodPlots(times, periods):
    plt.figure(figsize=winSize)

    plt.scatter(times, periods, color='b', s=4)
    plt.title("Best fit period for every quarter")
    plt.xlabel("Time (Days)")
    plt.ylabel("Best fit period (days)")

def produceQuarterFoldPlots(lightcurves, median_times, period, spacing, include_boot=False, n_samples=25):
    fig, ax = plt.subplots()
    fig.set(figheight=winSize[1], figwidth=winSize[0])

    ax.set_title("Folded light curves for every Kepler quarter")
    ax.set_xlabel("Phase")
    cmap = matplotlib.cm.get_cmap("winter")
    cmap_sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(median_times[0], median_times[-1]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)

    for idx, lc in enumerate(lightcurves):
        lc.fold(period)
        lc.sortToPhase()

        if include_boot:
            phase_samples, flux_samples = customBootstrap(lc.phase, lc.flux, n_samples)
            flux_samples = np.apply_along_axis(lambda x: getMedianLine(lc.phase, x, 2, sorted=True), 1, flux_samples)
            for p, f in zip(phase_samples, flux_samples):
                ax.plot(p, f-np.mean(f)-spacing*idx, alpha=0.3, c="orange")

        flux_binned = getMedianLine(lc.phase, lc.flux, 2, sorted=True)
        ax.plot(lc.phase, flux_binned-np.mean(flux_binned)-spacing*idx, c=cmap(idx/len(lightcurves)))

    fig.colorbar(cmap_sm, label="Median Quarter Time (Days)", cax=cax)

if __name__ == "__main__":

    # Initialize target constants
    target = "Kepler-78 b"
    if "Kepler" in target:
        target_period_range = getPeriodRange(sp_csv.loc[target, "pl_orbper"])
        target_duration = sp_csv.loc[target, "pl_trandur"]/24
    if np.isnan(target_duration):
        target_duration = 1/24
    else:
        target_period_range = None
        target_duration = 1/24
    quarters = np.linspace(0, 16, 17, dtype=int)
    lc = LC(np.empty(0), np.empty(0))
    filter_cutoff = 1

    periods = np.zeros_like(quarters, dtype=float)
    median_times = np.zeros_like(quarters, dtype=float)
    lc_list = np.zeros_like(quarters, dtype=object)

    for idx, quarter in enumerate(quarters):
        print("=" * 100)

        # Obtain data using lightkurve (Warning happens on following line)
        hdu = obtainFitsLightCurve(quarter=quarter, write_meta=False, target=target, mission="Kepler", exptime="long")
        quarter_lc = dataFromHDU(hdu, flux_column_name="FLUX")
        median_times[idx] = np.median(quarter_lc.time)
        print(f"Quarter: {quarter}")

        # Preprocess data by removing Nan values, and filtering out long term trends
        num_nans = fillNans(quarter_lc.flux)
        trend = filter(quarter_lc.flux, filter_cutoff, lowPassGaussian)
        quarter_lc.flux = quarter_lc.flux/trend - 1.0 
        quarter_lc.time, quarter_lc.flux, num_outliers = removeOutliers(quarter_lc.time, quarter_lc.flux, 8)
        print(f"{num_nans} nan flux values were filled in, {num_outliers} outliers were removed")
        print(f"Number of data points: {len(quarter_lc.time)}, Baseline: {quarter_lc.time[-1]-quarter_lc.time[0]} days")

        # Obtaining best fits periods
        """
        period = getPeriod(quarter_lc.time, quarter_lc.flux, target_duration, period=target_period_range)
        periods[idx] = period
        print(f"Period of {target} obtained from BLS periodogram: {period} Days or {period * 24} Hours")
        """

        # Appending the quarter specific lc for later use
        lc_list[idx] = quarter_lc
        lc.append(quarter_lc)

    print("=" * 100)

    print(f"Number of data points: {len(lc.time)}, Baseline: {lc.time[-1]-lc.time[0]} days")

    # Obtaining new data and constants with a variety of data analysis techniques
    period = getPeriod(lc.time, lc.flux, target_duration, period=target_period_range)
    spacing = 0.01 * (period ** 2 / (4.1 * 365))
    period_grid = np.arange(period-spacing*100, period+spacing*99, spacing)
    lc.fold(period)
    print(f"Period of {target} obtained from BLS periodogram: {period} Days or {period * 24} Hours")
    spacing = np.nanstd(lc.flux)*2

    # Producing a variety of informative plots and interactive plots
    produceBLSPeriodogramPlots(lc.time, lc.flux, kep78b_duration, period=target_period_range)
    produceFoldPlots(lc.time, lc.flux, period, time_bin_size=2)
    produceQuarterFoldPlots(lc_list, median_times, period, spacing, include_boot=True, n_samples=50)
    # produceQuarterPeriodPlots(median_times, periods)
    # produceFoldPlotsInteractive(lc.time, lc.flux, period_grid, target_duration)
    # produceFoldPlotsAnimation(lc.time, lc.flux, period_grid, target_duration)

    plt.show()
