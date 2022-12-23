import astropy
import lightkurve as lk
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy
import math
import pandas as pd
import copy

winSize = (10, 5)
np.set_printoptions(edgeitems=25)

# Dataframe with 1221 Short Period Exoplanets (P < 10 days)
sp_csv = pd.read_csv(filepath_or_buffer="ShortPeriodExoplanets.csv", comment="#")
sp_csv.set_index(keys="pl_name", inplace=True)
sp_csv.drop_duplicates(inplace=True)

# Kep78 Info, In Days
kep78b_baseline = 1424.89596
kep78b_period = 0.3550074
kep78b_duration = 0.0339167 

# Kepler-1520b Info, In Days
kep1520b_baseline = 1459.4893630790102
kep1520b_period = 0.6535555
kep1520b_duration = 0.0679167

# Class to encapsulate a stellar light curve, consisting of time data, flux data, and phase data if the light curve has been folded
class LC:
    def __init__(self, time, flux):
        self.time = time
        self.flux = flux
        self.data = {"time": self.time, "flux": flux}
        self.has_folded = False

    def __repr__(self):
        return f"Light Curve Object with data {self.data}"

    def plot(self, *args, **kwargs):
        plt.plot(self.time, self.flux, *args, **kwargs)
        plt.title("Light curve")
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Flux")

        plt.show()

    def getTime(self):
        return self.time

    def getFlux(self):
        return self.flux

    def zip(self):
        return zip(self.time, self.flux)
    
    def fold(self, period, epoch=0, copy=True):
        self.phase = fold(self.time, period, epoch=epoch)
        self.data["phase"] = self.phase
        self.has_folded = True
        if copy:
            out = self.copy()
            out.sortToPhase()
            return out

    def append(self, lc):
        self.time = np.append(self.time, lc.time)
        self.flux = np.append(self.flux, lc.flux)
    
    def sortToTime(self):
        sort_idx = np.argsort(self.time)
        self.time, self.flux = self.time[sort_idx], self.flux[sort_idx]

    def sortToPhase(self):
        if self.has_folded:
            sort_idx = np.argsort(self.phase)
            self.phase, self.flux = self.phase[sort_idx], self.flux[sort_idx]

    def copy(self):
        return copy.deepcopy(self)


def obtainFitsLightCurve(quarter=None, write_meta=False, *args, **kwargs):
    results = lk.search_lightcurve(*args, **kwargs)
    if write_meta: 
        results.table.write("DataTable.csv", format="ascii.csv", overwrite=True)
    if quarter is None:
        return results.download_all().stitch().to_fits(flux_column_name="SAP_FLUX")
    else:
        return results[quarter].download().to_fits(flux_column_name="SAP_FLUX")


def openFitsLightCurve(fits_file, include_image=False):
    with astropy.io.fits.open(name=fits_file, mode="readonly") as hdu:
        return dataFromHDU(hdu=hdu, include_image=include_image)


def dataFromHDU(hdu, include_image=False, flux_column_name="SAP_FLUX"):
    # Extracting the data from the HDUList
    data = hdu[1].data
    flux = data[flux_column_name]
    time = data['time']
    
    if include_image:
        image = hdu[2].data
        lc = LC(time, flux)
        lc.data["image"] = image
        return lc
    else:
        return LC(time, flux)


def getPeriodRange(period, baseline=4.1*365, buffer=1/24, low_buffer=None, up_buffer=None, spacing_coeff=0.01):
    # In Days
    if low_buffer is None:
        low_buffer = buffer
    if up_buffer is None:
        up_buffer = buffer
    spacing = spacing_coeff * (period  ** 2 / baseline) 
    return np.arange(period - low_buffer, period + up_buffer, spacing)  


def bin(y, bin_size=None, time_bin_size=None, weights=None, mode="median"):
    if bin_size is None and time_bin_size is None:
        bin_size = 47  # Adjust for data without a 30 minute cadence
    elif bin_size is None:
        bin_size = time_bin_size * 48 - 1  # Adjust for data without a 30 minute cadence
    if mode == "median":
        return scipy.signal.medfilt(y, kernel_size=bin_size)
    elif mode == "mean":
        if weights is None:
            weights = np.ones(bin_size) / bin_size
        elif len(weights) != bin_size:
            print("Warning, the length of the weights kernel isn't the same as bin_size")
        return np.convolve(np.pad(y, (bin_size // 2, bin_size // 2)), weights, mode="valid")
    else:
        print("Invalid mode for binning")


def fillNans(y):
    x = np.indices(y.shape)[0]
    isnan = np.isnan(y)
    nan_x = x[isnan].astype(int)
    nan_y = np.interp(nan_x, x[~isnan], y[~isnan])
    y[nan_x] = nan_y
    return len(nan_x)


def removeOutliers(x, y, n_sigma):
    num_outliers = 0
    mean = np.nanmean(y)
    std = np.nanstd(y)
    x_clean = np.empty(0)
    y_clean = np.empty(0)
    for x, y in zip(x, y):
        if np.abs(y - mean) < n_sigma * std:
            x_clean = np.append(x_clean, x)
            y_clean = np.append(y_clean, y)
        elif not np.isnan(y):
            num_outliers += 1
    return x_clean, y_clean, num_outliers


def customBootstrap(x, y, n_samples):
    x_samples = np.zeros((n_samples, x.shape[0]))
    y_samples = np.zeros((n_samples, y.shape[0]))
    sample_idx = np.indices(y.shape)[0]

    for idx in range(n_samples):
        resample_idx = np.sort(np.random.choice(a=sample_idx, size=y.shape))
        x_samples[idx] = x[resample_idx]
        y_samples[idx] = y[resample_idx]
    
    return x_samples, y_samples


def customResidualBootstrap(x, n_samples, block_size, smooth_func):
    sample = np.pad(x, (0, block_size), mode="median")

    x_smooth = smooth_func(sample)
    residuals = sample - x_smooth
    samples = np.zeros((n_samples, sample.shape[0]))

    for n in range(n_samples):
        for idx in range(len(residuals) - block_size):
            residual_block = residuals[idx:idx + block_size]
            smooth_block = x_smooth[idx:idx + block_size]
            samples[n][idx:idx + block_size] = smooth_block + np.random.choice(a=residual_block, size=residual_block.shape)
    
    return samples[:, :-block_size]


def lowPassGaussian(frequency, cutoff):
    gaussian = lambda x: math.e ** ((-0.5) * (x / cutoff) ** 2)
    return np.apply_along_axis(gaussian, 0, frequency)


def filter(flux, filter_type, cutoff):
    # Obtaining the applying the filter
    n = len(flux)
    frequency = scipy.fft.rfftfreq(n, 1/48)
    filter = filter_type(frequency, cutoff)
    fft_data = scipy.fft.rfft(flux)
    flux_filtered = scipy.fft.irfft(fft_data * filter, n)

    # Return the processed signal
    return flux_filtered


def produceTrendPlots(time, flux, filter_type, cutoff):
    plt.figure(figsize=winSize)

    # 1) Plotting the original SAP Light curves
    plt.subplot(2, 3, 1)
    plt.plot(time, flux)
    plt.title("Original SAP Light curve")
    plt.xlabel("Time (days)")
    plt.ylabel("Flux (electrons/second)")

    n = len(flux)
    intensity = scipy.fft.rfft(flux)
    frequency = scipy.fft.rfftfreq(n, 1/48)
    cutoff_idx = np.where(frequency > cutoff)[0][0]

    # 2) Plotting the fourier transform of the sap light curves
    plt.subplot(2, 3, 2)
    plt.plot(frequency, np.abs(intensity))
    plt.plot(cutoff, np.abs(intensity)[cutoff_idx], color="r", marker="o", markersize=10, label="Filter Cutoff")
    plt.axvline(cutoff, color="r")
    plt.title("Fourier Transform of SAP flux")
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Intensity")
    plt.legend()

    # 3) Plotting the filter
    filter = filter_type(frequency, cutoff)

    plt.subplot(2, 3, 3)
    plt.plot(frequency, filter)
    plt.title("Lowpass Box Filter Frequency Response")
    plt.plot(cutoff, filter[cutoff_idx], color="r", marker="o", markersize=10, label="Filter Cutoff")
    plt.axvline(cutoff, color="r")
    plt.xlabel("Frequency")
    plt.legend()

    # Filtering the light curves
    fft_data = scipy.fft.rfft(flux)
    flux_filtered = scipy.fft.irfft(fft_data * filter, n)

    # 4) Plotting the trend removed from the light curve
    plt.subplot(2, 3, 4)
    plt.plot(time, flux_filtered)
    plt.xlabel("Time (Days)")
    plt.ylabel("Flux (Normalized Units)")
    plt.title("Long Term Trend Removed From Light Curve")

    # 5) Plotting the low pass filtered light curve over the original light curve
    plt.subplot(2, 3, 5)
    plt.plot(time, flux, label="Before Filtering")
    plt.plot(time, flux_filtered, label="After Filtering")
    plt.title("Original and Low Pass Filtered Light Curves")
    plt.xlabel("Time (Days)")
    plt.ylabel("Flux (Normalized Units)")
    plt.legend()

    # 6) Plotting the original sap light curve divided by the filtered light curve
    plt.subplot(2, 3, 6)
    plt.plot(time, flux / flux_filtered - 1.0)
    plt.title("Low Pass Trend Removed")
    plt.xlabel("Time (Days)")
    plt.ylabel("Flux (Normalized Units)")

    # 6.5) Plotting the Savitzky-Golay Filtered Data
    """
    flux_filtered_savgol = scipy.signal.savgol_filter(x=flux, window_length=3, polyorder=0)
    plt.subplot(2, 3, 6)
    plt.plot(time, flux / flux_filtered_savgol)
    plt.title("Low Pass Trend Removed Using Savitzky-Golay")
    plt.xlabel("Time (Days)")
    plt.ylabel("Flux (Normalized Units)")
    """


def getPeriod(time, flux, duration, period=None):
    model = astropy.timeseries.BoxLeastSquares(time, flux)
    if period is None:
        period = model.autoperiod(duration=duration, frequency_factor=1)
    periodogram = model.power(period=period, duration=duration)
    return period[np.argmax(periodogram.power)]


def produceBLSPeriodogramPlots(time, flux, duration, period=None, is_78b=False):
    plt.figure(figsize=winSize)
    model = astropy.timeseries.BoxLeastSquares(time, flux)
    if period is None:
        period = model.autoperiod(duration=duration, frequency_factor=1)
    periodogram = model.power(period=period, duration=duration, objective="snr")
    best = period[np.argmax(periodogram.power)]

    # 1) Plotting the original SAP Light curves
    plt.subplot(1, 2, 1)
    plt.plot(time, flux, label="SAP FLUX")
    plt.plot(time, model.model(time, best, duration, periodogram.transit_time[0]), label="BLS Transit Model", color="r")
    plt.title("Light Curve")
    plt.xlabel("Time (Days)")
    plt.ylabel("Flux (Normalized Units)")
    plt.legend()

    # 2) Plotting the BLS periodogram
    plt.subplot(1, 2, 2)
    plt.plot(periodogram.period, periodogram.power, label="BLS Periodogram")
    if is_78b:
        plt.axvline(kep78b_period, color="r", label="Period of Kepler-78b from the Literature", linestyle="--")
    plt.title("Box Least Squares Periodogram")
    plt.xlabel("Period (Days)")
    plt.ylabel("BLS Power")
    plt.legend()

    return best


def fold(time, period, epoch=0):
    return (time - epoch) / period % 1


def produceFoldPlots(time, flux, period, bin_mode="median", time_bin_size=5):
    plt.figure(figsize=winSize)

    phase = fold(time, period)
    sort_index = np.argsort(phase)
    phase_sorted, flux_sorted = phase[sort_index], flux[sort_index]
    flux_binned = bin(flux_sorted, time_bin_size=time_bin_size, mode=bin_mode)

    plt.scatter(phase, flux, s=0.5, label="Phase folded light curve")
    plt.plot(phase_sorted, flux_binned, color="r", label="Binned phase folded light curve")
    plt.title(f"Phase folded light curve with period: {period}")
    plt.xlabel("Phase")
    plt.ylabel("Flux (Normalized Units)")
    

def produceFoldPlotsAnimation(time, flux, period_grid, duration, write=False):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    fig.suptitle("Phase Folded Light Curve and BLS Power With Varying Periods")

    plt.subplot(2, 1, 1)
    ax1.set_xlim(-0.2, 0.2)
    ax1.set_ylim(np.min(flux) - np.std(flux), np.max(flux) + np.std(flux))
    scatter, = ax1.plot([], [], linewidth=0, marker='o', markersize=0.5)
    plt.xlabel("Phase")
    plt.ylabel("Flux (Normalized Units)")

    model = astropy.timeseries.BoxLeastSquares(time, flux)
    periodogram = model.power(period=period_grid, duration=duration)

    plt.subplot(2, 1, 2)
    ax2.plot(periodogram.period, periodogram.power)
    plt.xlabel("Period (Days)")
    plt.ylabel("BLS Power")
    point, = ax2.plot([], [], marker='o', markersize=5, color='r')

    def init():
        scatter.set_data([], [])
        point.set_data([], [])
        return scatter, point,

    def animate(i):
        phase = fold(time, i)
        scatter.set_data(phase, flux)
        point.set_data([i], periodogram.power[np.where(period_grid == i)])
        return scatter, point,

    writer = matplotlib.animation.FFMpegWriter(fps=30)

    anim = matplotlib.animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=period_grid, interval=50, blit=True, repeat=False)

    if write:
        anim.save(filename="FoldedAnimation.mp4", writer=writer)

    plt.show()


def produceFoldPlotsInteractive(time, flux, period_grid, duration, initial_period=None):
    if initial_period is None:
        initial_period = period_grid[len(period_grid) // 2]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    fig.suptitle("Phase Folded Light Curve and BLS Power With Varying Periods")

    phase = fold(time, initial_period)

    plt.subplot(2, 1, 1)
    scatter, = ax1.plot(phase, flux, linewidth=0, marker='o', markersize=0.5)
    plt.xlabel("Phase")
    plt.ylabel("Flux (Normalized Units)")

    model = astropy.timeseries.BoxLeastSquares(t=time, y=flux)
    periodogram = model.power(period=period_grid, duration=duration)
    loc = periodogram.power[np.where(period_grid == initial_period)]

    plt.subplot(2, 1, 2)
    ax2.plot(periodogram.period, periodogram.power)
    plt.xlabel("Period (Days)")
    plt.ylabel("BLS Power")
    point, = ax2.plot([initial_period], loc, marker='o', markersize=5, color='r')

    def update(val):
        phase = fold(time, val)
        scatter.set_data(phase, flux)
        point.set_data([val], periodogram.power[np.where(period_grid == val)])
        fig.canvas.draw_idle()

    period_ax = plt.axes([0.13, 0.02, 0.75, 0.03])
    period_slider = matplotlib.widgets.Slider(ax=period_ax, label="Period", valmin=period_grid[0], valmax=period_grid[-1], valstep=period_grid, valinit=initial_period)
    period_slider.on_changed(update)

    plt.show()


def cut(time, flux, period): 
    time_cut = []
    flux_cut = []
    cut = 0
    for idx in range(1, len(time)):
        if time[idx] % period < time[idx - 1] % period:
            time_cut.append(time[cut:idx])
            flux_cut.append(flux[cut:idx])
            cut = idx

    time_cut = np.array(time_cut, dtype=object)
    flux_cut = np.array(flux_cut, dtype=object)
    return np.stack(arrays=(time_cut, flux_cut), axis=-1)


def produceSingleTransitPlotsAnimation(transits_cut, phase, flux, period, write=False):
    fig, ax = plt.subplots()
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    fig.suptitle("Varying transits plotted over the complete folded light curve")

    ax.set_xlim(np.min(phase) - np.std(phase), np.max(phase) + np.std(phase))
    ax.set_ylim(np.min(flux) - np.std(flux), np.max(flux) + np.std(flux))
    folded, = ax.plot([], [], lw=0, marker='o', markersize=0.5, label="Complete folded light curve")
    transit, = ax.plot([], [], color="r", label="Nth transit")
    plt.xlabel("Phase")
    plt.ylabel("Flux (Normalized Units)") 
    plt.legend()
 
    def init():
        folded.set_data(phase, flux)
        transit.set_data([], [])
        return folded, transit,

    def animate(i):
        transit_time, transit_flux = i
        transit_phase = fold(transit_time, period)
        folded.set_data(phase, flux)
        transit.set_data(transit_phase, transit_flux)
        return folded, transit,

    writer = matplotlib.animation.FFMpegWriter(fps=10)

    anim = matplotlib.animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=transits_cut, interval=20, blit=True)

    if write:
        anim.save(filename="TransitAnimation.mp4", writer=writer)

    plt.show()


def produceSingleTransitPlotsInteractive(transits_cut, phase, flux, period):
    fig, ax = plt.subplots()
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    fig.suptitle("Varying transits plotted over the complete folded light curve")

    n_transits = len(transits_cut)

    transit_time, transit_flux = transits_cut[0]
    transit_phase = fold(transit_time, period)
    folded, = ax.plot(phase, flux, lw=0, marker='o', markersize=0.5, label="Complete folded light curve")
    transit, = ax.plot(transit_phase, transit_flux, color="r", label="Nth transit")
    minimum, = ax.plot(transit_phase[np.argmin(transit_flux)], np.min(transit_flux), color='b', marker='o', markersize=4, label="Min",)
    maximum, = ax.plot(transit_phase[np.argmax(transit_flux)], np.max(transit_flux), color='g', marker='o', markersize=4, label="Max")
    plt.xlabel("Phase")
    plt.ylabel("Flux (Normalized Units)") 
    plt.legend()

    def update(val):
        transit_time, transit_flux = transits_cut[val]
        transit_phase = fold(transit_time, period)
        folded.set_data(phase, flux)
        transit.set_data(transit_phase, transit_flux)
        minimum.set_data(transit_phase[np.argmin(transit_flux)], np.min(transit_flux))
        maximum.set_data(transit_phase[np.argmax(transit_flux)], np.max(transit_flux))
        fig.canvas.draw_idle()

    transit_ax = plt.axes([0.13, 0.02, 0.75, 0.03])
    transit_slider = matplotlib.widgets.Slider(ax=transit_ax, label="Number transit", valmin=0, valmax=n_transits - 1, valstep=np.arange(0, n_transits), valinit=0)
    transit_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":

    """
    Target List:
    target: Kepler-78b, notes: primary target for analysis
    target: Kepler-1520b, notes: disintegrating planet
    Kepler-41 b, notes: hot Jupiter
    Kepler-12 b, notes: hot Jupiter
    Kepler-17 b, notes: transit timing variations
    Kepler-7 b, notes: hot Jupiter
    Kepler-42 c, notes: extremely short period
    """

    # Initialize Target Constants
    target = "Kepler-78 b"
    if target in sp_csv.index:
        target_period_range = getPeriodRange(period=sp_csv.loc[target, "pl_orbper"])
        target_duration = sp_csv.loc[target, "pl_trandur"] / 24
        if np.isnan(target_duration):
            print("Target duration not found in provided CSV file")
            target_duration = 1/24
    else:
        print("Warning, target not found in provided CSV file")
        target_period_range = None
        target_duration = 1/24
    quarter = 2
    filter_cutoff = 1

    print("=" * 100)

    # Obtain data using lightkurve (Warning happens on following line)
    hdu = obtainFitsLightCurve(quarter=quarter, target=target, mission="Kepler", exptime="long")
    lc = dataFromHDU(hdu=hdu, flux_column_name="FLUX")
    print(f"Quarter: {quarter}")

    # Preprocess data by removing Nan values, and filtering out long term trends
    num_nans = fillNans(lc.flux)
    produceTrendPlots(time=lc.time, flux=lc.flux, filter_type=lowPassGaussian, cutoff=filter_cutoff)
    trend = filter(lc.flux, filter_type=lowPassGaussian, cutoff=filter_cutoff)
    lc.flux = lc.flux/trend - 1.0
    lc.time, lc.flux, num_outliers = removeOutliers(lc.time, lc.flux, n_sigma=8)
    print(f"{num_nans} nan flux values were filled in, {num_outliers} outliers were removed")
    print(f"Number of data points: {len(lc.time)}, Baseline: {lc.time[-1] - lc.time[0]} days")

    # Obtaining new data and constants with a variety of data analysis techniques
    period = getPeriod(lc.time, lc.flux, duration=target_duration, period=target_period_range)
    period_grid = getPeriodRange(period=period, buffer=1/(24*60))
    lc_folded = lc.fold(period)
    lc_folded.data["test"] = "test"
    transits_cut = cut(lc.time, lc.flux, period)
    print(f"Period of {target} obtained from BLS periodogram: {period} Days or {period * 24} Hours")

    print("=" * 100)

    # Producing a variety of informative plots and interactive plots
    produceBLSPeriodogramPlots(time=lc.time, flux=lc.flux, duration=target_duration, period=target_period_range)
    produceFoldPlots(time=lc.time, flux=lc.flux, period=period, bin_mode="median", time_bin_size=1)
    # produceFoldPlotsInteractive(time=lc.time, flux=lc.flux, period_grid=period_grid, duration=target_duration)
    # produceSingleTransitPlotsInteractive(transits_cut=transits_cut[:100], phase=lc.phase, flux=lc.flux, period=period)
    # produceFoldPlotsAnimation(lc.time, lc.flux, period_grid, target_duration)
    # produceSingleTransitPlotsAnimation(transits_cut=transits_cut[:100], phase=lc.phase, flux=lc.flux, period=period)

    # Bootstrap Test
    # ===============================================================================================
    # Instantiating the figure and pre-processing the folded transit data
    n_samples = 100
    fig, (ax1) = plt.subplots(1, 1)
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    ax1.set_title(f"Bootstrapped Folded Light Curve ({n_samples} samples)")  
    # ax2.set_title("Residuals Used for Bootstrapping")  
    smooth_func = lambda x: filter(x, filter_type=lowPassGaussian, cutoff=filter_cutoff)

    # Preforming residual resampling with a custom function
    """
    samples = customResidualBootstrap(lc_folded.flux, n_samples=n_samples, block_size=10, smooth_func=smooth_func)
    """

    # Preforming classic resampling with a custom function
    phase_samples, samples = customBootstrap(lc_folded.phase, lc_folded.flux, n_samples=n_samples)

    # Plotting the original folded transit data and the bootstrap samples on top
    flux_binned = bin(lc_folded.flux, mode="median", time_bin_size=1)
    smooth_data = smooth_func(lc_folded.flux)
    samples = np.apply_along_axis(lambda x: bin(x, mode="median", time_bin_size=1), 1, samples)
    upper = np.apply_along_axis(lambda x: np.percentile(x, 95), 0, samples)
    lower = np.apply_along_axis(lambda x: np.percentile(x, 5), 0, samples)

    ax1.scatter(lc_folded.phase, lc_folded.flux, s=2)
    # ax1.fill_between(phase, lower, upper, alpha=0.8, color="orange")
    for p, f in zip(phase_samples, samples):
        ax1.plot(p, f, alpha=0.3, color="orange")
    ax1.plot(lc_folded.phase, flux_binned, color="r")
    # ax1.plot(phase, smooth_data, color="g")
    # ax2.scatter(lc_folded.phase, lc_folded.flux-smooth_data, s=2)
    # ===============================================================================================

    plt.show()
    