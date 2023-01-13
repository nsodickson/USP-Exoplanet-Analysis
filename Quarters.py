from Pipeline import *

def produceQuarterPeriodPlots(time, period_list):
    plt.figure(figsize=winSize)

    plt.scatter(time, period_list, color='b', s=4)
    plt.title("Best fit period for every quarter")
    plt.xlabel("Time (Days)")
    plt.ylabel("Best fit period (days)")

def produceTimeBinPlots(time, lc_list, period, spacing, include_boot=False, n_samples=25, **kwargs):
    fig, ax = plt.subplots()
    fig.set(figheight=winSize[1], figwidth=winSize[0])

    ax.set_title("Folded light curves for every Kepler quarter")
    ax.set_xlabel("Phase")
    cmap = matplotlib.cm.get_cmap("winter")
    cmap_sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(time[0], time[-1]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    # Pipeline to apply to every quarter
    for i, lc in enumerate(lc_list):
        lc.fold(period).sortToPhase()

        if include_boot:
            phase_samples, flux_samples = customBootstrap(lc.phase, lc.flux, n_samples=n_samples)
            for p, f in zip(phase_samples, flux_samples):
                p_binned, f_binned = timeBin(p, f, **kwargs)
                ax.plot(p_binned, f_binned - np.mean(f_binned) - spacing * i, alpha=0.3, c="orange")

        phase_binned, flux_binned = timeBin(lc.phase, lc.flux, **kwargs)
        ax.plot(phase_binned, flux_binned - np.mean(flux_binned) - spacing * i, c=cmap(i / len(lc_list)))

    fig.colorbar(cmap_sm, label="Median Quarter Time (Days)", cax=cax)

def produceFrequencyBinPlots(time, lc_list, period, spacing, frequencies, **kwargs):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set(figheight=winSize[1], figwidth=winSize[0])

    ax1.set_title("Folded Light Curves Weighted With Varying Frequencies")
    ax2.set_title("Sine and Cosine curves used as weights")
    ax1.set_xlabel("Phase")
    ax2.set_xlabel("Phase")

    lc_avg = 0
    for n, lc in enumerate(lc_list):
        lc.fold(period)
        phase_binned, flux_binned = timeBin(lc.phase, lc.flux, **kwargs)
        lc_avg += flux_binned
    lc_avg /= len(lc_list)
    ax1.plot(phase_binned, lc_avg, c="g")
    ax2.plot(time, np.ones_like(time), color="g")

    for i, f in enumerate(frequencies):
        idx = i + 1
        lc_avg_sin = 0
        lc_avg_cos = 0
        for n, lc in enumerate(lc_list):
            phase_binned, flux_binned = timeBin(lc.phase, lc.flux, **kwargs)
            lc_avg_sin += flux_binned * np.sin(time[n] * f)
            lc_avg_cos += flux_binned * np.cos(time[n] * f)
        ax1.plot(phase_binned, lc_avg_sin - np.mean(lc_avg_sin) - spacing * idx, c="r")
        ax1.plot(phase_binned, lc_avg_cos - np.mean(lc_avg_cos) - spacing * idx, c="b")
        ax2.plot(time, np.sin(time * f) - 3 * idx, color="r")
        ax2.plot(time, np.cos(time * f) - 3 * idx, color="b")

        
if __name__ == "__main__":

    # Initialize target constants
    target = "Kepler-78 b"
    if target in sp_csv.index:
        target_period_range = getPeriodRange(sp_csv.loc[target, "pl_orbper"])
        target_duration = sp_csv.loc[target, "pl_trandur"] / 24
        if np.isnan(target_duration):
            print("Target duration not found in provided CSV file")
            target_duration = 1/24
    else:
        print("Warning, target not found in provided CSV file")
        target_period_range = None
        target_duration = 1/24
    quarters = np.linspace(0, 16, 17, dtype=int)
    lc = LC(np.empty(0), np.empty(0))
    filter_cutoff = 1

    period_list = np.zeros_like(quarters, dtype=float)
    time_list = np.zeros_like(quarters, dtype=float)
    lc_list = np.zeros_like(quarters, dtype=object)

    # Pipeline to apply to every quarter
    for idx, quarter in enumerate(quarters):
        print("=" * 100)

        # Obtain data using lightkurve (Warning happens on following line)
        hdu = obtainFitsLightCurve(quarter=quarter, target=target, mission="Kepler", exptime="long")
        quarter_lc = dataFromHDU(hdu=hdu, flux_column_name="FLUX")
        time_list[idx] = np.median(quarter_lc.time)
        print(f"Quarter: {quarter}")

        # Preprocess data by removing Nan values, and filtering out long term trends
        num_nans = fillNans(quarter_lc.flux)
        trend = filter(quarter_lc.flux, lowPassGaussian, filter_cutoff)
        quarter_lc.flux = quarter_lc.flux / trend - 1.0 
        quarter_lc.time, quarter_lc.flux, num_outliers = removeOutliers(quarter_lc.time, quarter_lc.flux, n_sigma=8)
        print(f"{num_nans} nan flux values were filled in, {num_outliers} outliers were removed")
        print(f"Number of data points: {len(quarter_lc.time)}, Baseline: {quarter_lc.time[-1] - quarter_lc.time[0]} days")

        # Obtaining best fits periods
        """
        period = getPeriod(quarter_lc.time, quarter_lc.flux, target_duration, period=target_period_range)
        period_list[idx] = period
        print(f"Period of {target} obtained from BLS periodogram: {period} Days or {period * 24} Hours")
        """

        # Appending the quarter specific lc for later use
        lc_list[idx] = quarter_lc
        lc.append(quarter_lc)

    print("=" * 100)

    baseline = lc.time[-1] - lc.time[0]
    print(f"Number of data points: {len(lc.time)}, Baseline: {baseline} days")

    # Obtaining new data and constants with a variety of data analysis techniques
    period = getPeriod(lc.time, lc.flux, duration=target_duration, period=target_period_range)
    period_grid = getPeriodRange(period=period, buffer=1 / (24 * 60))
    lc.fold(period)
    print(f"Period of {target} obtained from BLS periodogram: {period} Days or {period * 24} Hours")
    # print(f"Standard deviation of periods with quarter 0: {np.std(periods)}, without quarter 0: {np.std(periods[1:])}")
    spacing = np.nanstd(lc.flux) * 2.5

    frequencies = [i * (2 * math.pi / baseline) for i in range(1, 17)]

    # Producing a variety of informative plots and interactive plots
    produceBLSPeriodogramPlots(time=lc.time, flux=lc.flux, duration=target_duration, period=target_period_range)
    produceFoldPlots(time=lc.time, flux=lc.flux, period=period, include_binned=True, bins=np.arange(0, 1.01, 0.01), aggregate_func=np.mean)
    # produceTimeBinPlots(time=time_list, lc_list=lc_list, period=period, spacing=spacing, include_boot=True, n_samples=50, bins=np.arange(0, 1.01, 0.01), aggregate_func=np.mean)
    produceFrequencyBinPlots(time=time_list, lc_list=lc_list, period=period, spacing=spacing * 4, frequencies=frequencies, bins=np.arange(0, 1.01, 0.01), aggregate_func=np.mean)
    # produceQuarterPeriodPlots(time=time_list, period_list=period_list)
    # produceFoldPlotsInteractive(time=lc.time, flux=lc.flux, period_grid=period_grid, duration=target_duration)
    # produceFoldPlotsAnimation(time=lc.time, flux=lc.flux, period_grid=period_grid, duration=target_duration)

    plt.show()
