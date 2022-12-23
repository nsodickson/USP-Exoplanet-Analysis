from Pipeline import *

def produceQuarterPeriodPlots(time_list, period_list):
    plt.figure(figsize=winSize)

    plt.scatter(time_list, period_list, color='b', s=4)
    plt.title("Best fit period for every quarter")
    plt.xlabel("Time (Days)")
    plt.ylabel("Best fit period (days)")

def produceQuarterFoldPlots(lc_list, time_list, period, spacing, bin_mode="median", time_bin_size=5, include_boot=False, n_samples=25):
    fig, ax = plt.subplots()
    fig.set(figheight=winSize[1], figwidth=winSize[0])

    ax.set_title("Folded light curves for every Kepler quarter")
    ax.set_xlabel("Phase")
    cmap = matplotlib.cm.get_cmap("winter")
    cmap_sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(time_list[0], time_list[-1]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    # Pipeline to apply to every quarter
    for idx, lc in enumerate(lc_list):
        lc = lc.fold(period, copy=False)
        lc.sortToPhase()

        if include_boot:
            phase_samples, flux_samples = customBootstrap(lc.phase, lc.flux, n_samples=n_samples)
            flux_samples = np.apply_along_axis(lambda x: bin(x, time_bin_size=time_bin_size, mode=bin_mode), 1, flux_samples)
            for p, f in zip(phase_samples, flux_samples):
                ax.plot(p, f - np.mean(f) - spacing * idx, alpha=0.3, c="orange")

        flux_binned = bin(lc.flux, time_bin_size=time_bin_size, mode=bin_mode)
        ax.plot(lc.phase, flux_binned - np.mean(flux_binned) - spacing * idx, c=cmap(idx / len(lc_list)))

    fig.colorbar(cmap_sm, label="Median Quarter Time (Days)", cax=cax)

def produceFrequencyBinPlots(lc, spacing, frequencies, time_bin_size=5):
    # Precondition: lc is folded
    fig, ax = plt.subplots()
    fig.set(figheight=winSize[1], figwidth=winSize[0])

    ax.set_title("Folded Light Curves Weighted With Varying Angular Frequencies")
    ax.set_xlabel("Phase")
    cmap = matplotlib.cm.get_cmap("winter")
    cmap_sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(frequencies[0], frequencies[-1]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    for idx, f in enumerate(frequencies):
        # Code where weighted mean takes place
        pass
        
    fig.colorbar(cmap_sm, label="Frequency", cax=cax)

if __name__ == "__main__":

    # Initialize target constants
    target = "Kepler-1520 b"
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
        hdu = obtainFitsLightCurve(quarter=quarter, write_meta=False, target=target, mission="Kepler", exptime="long")
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
    lc_folded = lc.fold(period)
    transits_cut = cut(lc.time, lc.flux, period)
    print(f"Period of {target} obtained from BLS periodogram: {period} Days or {period * 24} Hours")
    # print(f"Standard deviation of periods with quarter 0: {np.std(periods)}, without quarter 0: {np.std(periods[1:])}")
    spacing = np.nanstd(lc.flux) * 2.5    

    frequencies = [2 * math.pi / (baseline / i) for i in range(1, 16)]

    # Producing a variety of informative plots and interactive plots
    # produceFrequencyBinPlots(lc_folded, spacing, frequencies)
    # produceBLSPeriodogramPlots(time=lc.time, flux=lc.flux, duration=target_duration, period=target_period_range)
    # produceFoldPlots(time=lc.time, flux=lc.flux, period=period, bin_mode="mean", time_bin_size=1)
    # produceQuarterFoldPlots(lightcurves=lc_list, time_list=time_list, period=period, spacing=spacing, bin_mode="mean", time_bin_size=1, include_boot=True, n_samples=50)
    # produceQuarterPeriodPlots(time_list=time_list, period_list=period_list)
    # produceFoldPlotsInteractive(time=lc.time, flux=lc.flux, period_grid=period_grid, duration=target_duration)
    # produceSingleTransitPlotsInteractive(transits_cut=transits_cut[:100], phase=lc.phase, flux=lc.flux, period=period)
    # produceFoldPlotsAnimation(time=lc.time, flux=lc.flux, period_grid=period_grid, duration=target_duration)
    # produceSingleTransitPlotsAnimation(transits_cut=transits_cut[:100], phase=lc.phase, flux=lc.flux, period=period)

    plt.show()
