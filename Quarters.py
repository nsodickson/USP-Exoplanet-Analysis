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
        lc.fold(period).sortToPhase()

        if include_boot:
            phase_samples, flux_samples = customBootstrap(lc.phase, lc.flux, n_samples=n_samples)
            flux_samples = np.apply_along_axis(lambda x: bin(x, time_bin_size=time_bin_size, mode=bin_mode), 1, flux_samples)
            for p, f in zip(phase_samples, flux_samples):
                ax.plot(p, f - np.mean(f) - spacing * idx, alpha=0.3, c="orange")

        flux_binned = bin(lc.flux, time_bin_size=time_bin_size, mode=bin_mode)
        ax.plot(lc.phase, flux_binned - np.mean(flux_binned) - spacing * idx, c=cmap(idx / len(lc_list)))

    fig.colorbar(cmap_sm, label="Median Quarter Time (Days)", cax=cax)

def produceFrequencyBinPlots(lc, spacing, frequencies, time_bin_size=5):
    # Precondition: lc is folded on the best fit period
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set(figheight=winSize[1], figwidth=winSize[0])

    bin_size = time_bin_size * 48 - 1  # Adjust for differently spaced data (currently 30 minute cadence)

    ax1.set_title("Folded Light Curves Weighted With Varying Frequencies")
    ax2.set_title("Sine and Cosine curves used as weights")
    ax1.set_xlabel("Phase")
    ax2.set_xlabel("Phase")

    for idx, f in enumerate(frequencies):
        # Test 1: Calculating the weighted moving average of the flux values clamping the kernel at the edges so the shape stays the same
        """
        flux_avg_sin = np.zeros_like(lc.phase)
        flux_avg_cos = np.zeros_like(lc.phase)
        for i, flux in enumerate(lc.flux):
            low = max(0, i - bin_size // 2)  # Clamps the kernel at the beginning of the signal 
            up = min(len(lc.flux), i + bin_size // 2)  # Clamps the kernel at the end of the signal
            flux_avg_sin[i] = np.average(lc.flux[low:up], weights=np.sin(lc.phase[low:up] * f) + 1)  # +1 to prevent weights that add to zero, temporary solution
            flux_avg_cos[i] = np.average(lc.flux[low:up], weights=np.cos(lc.phase[low:up] * f) + 1)  # +1 to prevent weights that add to zero, temporary solution
        # Ploting the weighted averages
        ax1.plot(lc.phase, flux_avg_sin - spacing * idx, c="r")
        ax1.plot(lc.phase, flux_avg_cos - spacing * idx, c="b")
        # Plotting the weights 
        ax2.plot(lc.phase, np.sin(lc.phase * f) - 3 * idx, c="r")
        ax2.plot(lc.phase, np.cos(lc.phase * f) - 3 * idx, c="b")
        """
        
        # Test 2: Calculating the weighted moving average of the flux values cutting off values on the edges
        """
        flux_avg_sin = np.zeros(len(lc.phase) - bin_size + 1)
        flux_avg_cos = np.zeros(len(lc.phase) - bin_size + 1)
        for i in range(bin_size // 2, len(lc.flux) - bin_size // 2):
            low = i - bin_size // 2  
            up = i + bin_size // 2
            flux_avg_sin[i - bin_size // 2] = np.average(lc.flux[low:up], weights=np.sin(lc.phase[low:up] * f) + 1)  # +1 to prevent weights that add to zero, temporary solution
            flux_avg_cos[i - bin_size // 2] = np.average(lc.flux[low:up], weights=np.cos(lc.phase[low:up] * f) + 1)  # +1 to prevent weights that add to zero, temporary solution
        # Plotting the weighted averages, cutting of the phase values in the same way the flux values were cut off
        ax1.plot(lc.phase[bin_size // 2:-bin_size // 2 + 1], flux_avg_sin - spacing * idx, c="r")
        ax1.plot(lc.phase[bin_size // 2:-bin_size // 2 + 1], flux_avg_cos - spacing * idx, c="b")
        # Plotting the weights 
        ax2.plot(lc.phase, np.sin(lc.phase * f) - 3 * idx, c="r")
        ax2.plot(lc.phase, np.cos(lc.phase * f) - 3 * idx, c="b")
        """
        
        # Test 3: Calculating the weighted moving average of the flux values padding the flux and phase values
        """
        flux_padded = np.pad(lc.flux, (bin_size // 2, bin_size // 2))
        phase_padded = np.pad(lc.phase, (bin_size // 2, bin_size // 2))
        flux_avg_sin = np.zeros_like(lc.phase)
        flux_avg_cos = np.zeros_like(lc.phase)
        for i in range(len(lc.flux)):
            up = i + bin_size
            flux_avg_sin[i] = np.average(flux_padded[i:up], weights=np.sin(phase_padded[i:up] * f) + 1)  # +1 to prevent weights that add to zero, temporary solution
            flux_avg_cos[i] = np.average(flux_padded[i:up], weights=np.cos(phase_padded[i:up] * f) + 1)  # +1 to prevent weights that add to zero, temporary solution
        # Plotting the weighted averages
        ax1.plot(lc.phase, flux_avg_sin - spacing * idx, c="r")
        ax1.plot(lc.phase, flux_avg_cos - spacing * idx, c="b")
        # Plotting the weights
        ax2.plot(lc.phase, np.sin(lc.phase * f) - 3 * idx, c="r")
        ax2.plot(lc.phase, np.cos(lc.phase * f) - 3 * idx, c="b")
        """
        

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
    lc_folded = lc.fold(period).copy().sortToPhase()
    print(f"Period of {target} obtained from BLS periodogram: {period} Days or {period * 24} Hours")
    # print(f"Standard deviation of periods with quarter 0: {np.std(periods)}, without quarter 0: {np.std(periods[1:])}")
    spacing = np.nanstd(lc.flux) * 2.5

    frequencies = [2 * math.pi * i for i in range(25)]

    # Producing a variety of informative plots and interactive plots
    produceFrequencyBinPlots(lc_folded, spacing, frequencies, time_bin_size=1)
    # produceBLSPeriodogramPlots(time=lc.time, flux=lc.flux, duration=target_duration, period=target_period_range)
    # produceFoldPlots(time=lc.time, flux=lc.flux, period=period, bin_mode="mean", time_bin_size=1)
    # produceQuarterFoldPlots(lc_list=lc_list, time_list=time_list, period=period, spacing=spacing, bin_mode="mean", time_bin_size=1, include_boot=True, n_samples=50)
    # produceQuarterPeriodPlots(time_list=time_list, period_list=period_list)
    # produceFoldPlotsInteractive(time=lc.time, flux=lc.flux, period_grid=period_grid, duration=target_duration)
    # produceFoldPlotsAnimation(time=lc.time, flux=lc.flux, period_grid=period_grid, duration=target_duration)

    plt.show()
