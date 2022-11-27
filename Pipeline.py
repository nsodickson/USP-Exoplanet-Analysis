import astropy
import lightkurve as lk
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import scipy
from tsmoothie import bootstrap
from tsmoothie import smoother
from tsmoothie.utils_func import _check_data, _check_output
import math
import pandas as pd

winSize = (10, 5)
np.set_printoptions(edgeitems=25)

# Dataframe with 1221 Short Period Exoplanets (P < 10 days)
sp_csv = pd.read_csv("ShortPeriodExoplanets.csv")
sp_csv.set_index("pl_name", inplace=True)
sp_csv.drop_duplicates(inplace=True)

# Kep78 Info, In Days
kep78b_baseline = 1424.89596
kep78b_period = 0.3550074
kep78b_duration = 0.0339167 

# Kepler-1520b Info, In Days
kep1520b_baseline = 1459.4893630790102
kep1520b_period = 0.6535555
kep1520b_duration = 0.0679167


class LC:
    def __init__(self, time, flux, time_unit="days"):
        self.time = time
        self.flux = flux
        self.data = {"time": self.time, "flux": flux}
        self.time_unit = time_unit
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
    
    def fold(self, period):
        self.phase = fold(self.time, period)
        self.data["phase"] = self.phase
        self.has_folded = True

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


def getPeriodRange(period, baseline=4.1*365, buffer=1/24, low_buffer=None, up_buffer=None):
    # In Days
    if low_buffer is None:
        low_buffer = buffer
    if up_buffer is None:
        up_buffer = buffer
    spacing = 0.01 * (period ** 2 / (baseline)) 
    return np.arange((period - low_buffer), (period + up_buffer), spacing)  


def fillNans(y_data):
    x_data = np.indices(y_data.shape)[0]
    isnan = np.isnan(y_data)
    nan_x = x_data[isnan].astype(int)
    nan_y = np.interp(nan_x, x_data[~isnan], y_data[~isnan])
    y_data[nan_x] = nan_y
    return len(nan_x)


def removeOutliers(x_data, y_data, n_sigma):
    num_outliers = 0
    mean = np.nanmean(y_data)
    std = np.nanstd(y_data)
    x_data_clean = np.empty(0)
    y_data_clean = np.empty(0)
    for x, y in zip(x_data, y_data):
        if np.abs(y-mean) < n_sigma * std:
            x_data_clean = np.append(x_data_clean, x)
            y_data_clean = np.append(y_data_clean, y)
        elif not np.isnan(y):
            num_outliers += 1
    return x_data_clean, y_data_clean, num_outliers


def lowPassGaussian(freq, freq_cutoff):
    gaussian = lambda x: math.e ** ((-1 / 2) * (x / freq_cutoff) ** 2)
    return np.apply_along_axis(gaussian, 0, freq)


def obtainFitsLightCurve(quarter=None, write_meta=True, *args, **kwargs):
    results = lk.search_lightcurve(*args, **kwargs)
    if write_meta: 
        results.table.write("DataTable.csv", format="ascii.csv", overwrite=True)
    if quarter is None:
        return results.download_all().stitch().to_fits(flux_column_name="SAP_FLUX")
    else:
        return results[quarter].download().to_fits(flux_column_name="SAP_FLUX")


def dataFromHDU(hdu: astropy.io.fits.HDUList, include_image=False, flux_column_name="SAP_FLUX"):
    # Extracting the data from the HDUList
    b = hdu[1].data
    h = hdu[1].header
    flux = b[flux_column_name]
    time = b['time']
    
    if include_image:
        image = hdu[2].data
        lc = LC(time, flux)
        lc.data["image"] = image
        return lc
    else:
        return LC(time, flux)


def openFitsLightCurve(fits_file: str, include_image=False):
    with astropy.io.fits.open(fits_file, mode="readonly") as hdu:
        return dataFromHDU(hdu, include_image=include_image)


def filter(flux, cutoff, filter_type):
    # Obtaining the applying the filter
    n = len(flux)
    freq = scipy.fft.rfftfreq(n, 1/48)
    filter = filter_type(freq, cutoff)
    fft_data = scipy.fft.rfft(flux)
    flux_filtered = scipy.fft.irfft(fft_data*filter, n)

    # Return the processed signal
    return flux_filtered


def produceTrendPlots(time, flux, cutoff, filter_type):
    plt.figure(figsize=winSize)

    # 1) Plotting the original SAP Light curves
    plt.subplot(2, 3, 1)
    plt.plot(time, flux)
    plt.title("Original SAP Light curve")
    plt.xlabel("Time (days)")
    plt.ylabel("Flux (electrons/second)")

    n = len(flux)
    step = 1 / 48  # There are 48 observations per day for long cadence light curves
    intensity = scipy.fft.rfft(flux)
    frequency = scipy.fft.rfftfreq(n, step)
    cutoff_idx = np.where(frequency > cutoff)[0][0]

    # 2) Plotting the fourier transform of the sap light curves (excluding the zero frequency)
    plt.subplot(2, 3, 2)
    plt.plot(frequency, np.abs(intensity))
    plt.plot(cutoff, np.abs(intensity)[cutoff_idx], color="r", marker="o", ms=10, label="Filter Cutoff")
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
    plt.plot(cutoff, filter[cutoff_idx], color="r", marker="o", ms=10, label="Filter Cutoff")
    plt.axvline(cutoff, color="r")
    plt.xlabel("Frequency")
    plt.legend()

    # Filtering and padding the light curves
    fft_data = scipy.fft.rfft(flux)
    flux_filtered = scipy.fft.irfft(fft_data*filter, n)

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
    plt.plot(time, flux/flux_filtered - 1.0)
    plt.title("Low Pass Trend Removed")
    plt.xlabel("Time (Days)")
    plt.ylabel("Flux (Normalized Units)")

    # 6.5) Plotting the Savitzky-Golay Filtered Data
    """
    flux_filtered_savgol = scipy.signal.savgol_filter(flux, 3, 0)
    plt.subplot(2, 3, 6)
    plt.plot(time, flux/flux_filtered_savgol)
    plt.title("Low Pass Trend Removed Using Savitzky-Golay")
    plt.xlabel("Time (Days)")
    plt.ylabel("Flux (Normalized Units)")
    """


def getPeriod(time, flux, duration, period=None):
    model = astropy.timeseries.BoxLeastSquares(time, flux)
    if period is None:
        period = model.autoperiod(duration, frequency_factor=1)
    periodogram = model.power(period, duration)
    return period[np.argmax(periodogram.power)]


def produceBLSPeriodogramPlots(time, flux, duration, period=None, is_78b=False):
    plt.figure(figsize=winSize)
    model = astropy.timeseries.BoxLeastSquares(time, flux)
    if period is None:
        period = model.autoperiod(duration, frequency_factor=1)
    periodogram = model.power(period, duration, objective="snr")
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


def fold(time, period):
    return time % period - period/2


def getMedianLine(phase, flux, time_bin_size, sorted=False):
    if not sorted:
        sort_index = np.argsort(phase)
        phase_sorted, flux_sorted = phase[sort_index], flux[sort_index]
        return phase_sorted, scipy.signal.medfilt(flux_sorted, kernel_size=time_bin_size*24*2-1)
    else:
        return scipy.signal.medfilt(flux, kernel_size=time_bin_size*24*2-1)


def produceFoldPlots(time, flux, period, time_bin_size=5):
    # time_bin_size is In Days, Approximately

    plt.figure(figsize=winSize)

    phase = fold(time, period)
    phase_sorted, flux_binned = getMedianLine(phase, flux, time_bin_size)

    plt.scatter(phase, flux, s=0.5, label="Phase folded light curve")
    plt.plot(phase_sorted, flux_binned, color="r", label="Binned phase folded light curve")
    plt.title(f"Phase folded light curve with period: {period}")
    plt.xlabel("Phase")
    plt.ylabel("Flux (Normalized Units)")
    

def produceFoldPlotsAnimation(time, flux, period_range, duration, write=False):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    fig.suptitle("Phase Folded Light Curve and BLS Power With Varying Periods")

    plt.subplot(2, 1, 1)
    ax1.set_xlim(-0.2, 0.2)
    ax1.set_ylim(np.min(flux)-np.std(flux), np.max(flux)+np.std(flux))
    scatter, = ax1.plot([], [], linewidth=0, marker='o', markersize=0.5)
    plt.xlabel("Phase")
    plt.ylabel("Flux (Normalized Units)")

    model = astropy.timeseries.BoxLeastSquares(time, flux)
    periodogram = model.power(period_range, duration)

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
        point.set_data([i], periodogram.power[np.where(period_range == i)])
        return scatter, point,

    writer = matplotlib.animation.FFMpegWriter(fps=30)

    anim = matplotlib.animation.FuncAnimation(fig, func=animate, init_func=init, frames=period_range, interval=50, blit=True, repeat=False)

    if write:
        anim.save("FoldedAnimation.mp4", writer)

    plt.show()


def produceFoldPlotsInteractive(time, flux, period_range, duration, initial_period=None):
    if initial_period is None:
        initial_period = period_range[len(period_range) // 2]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    fig.suptitle("Phase Folded Light Curve and BLS Power With Varying Periods")

    phase = fold(time, initial_period)

    plt.subplot(2, 1, 1)
    scatter, = ax1.plot(phase, flux, linewidth=0, marker='o', markersize=0.5)
    plt.xlabel("Phase")
    plt.ylabel("Flux (Normalized Units)")

    model = astropy.timeseries.BoxLeastSquares(time, flux)
    periodogram = model.power(period_range, duration)
    loc = periodogram.power[np.where(period_range == initial_period)]

    plt.subplot(2, 1, 2)
    ax2.plot(periodogram.period, periodogram.power)
    plt.xlabel("Period (Days)")
    plt.ylabel("BLS Power")
    point, = ax2.plot([initial_period], loc, marker='o', markersize=5, color='r')

    def update(val):
        phase = fold(time, val)
        scatter.set_data(phase, flux)
        point.set_data([val], periodogram.power[np.where(period_range == val)])
        fig.canvas.draw_idle()

    period_ax = plt.axes([0.13, 0.02, 0.75, 0.03])
    period_slider = matplotlib.widgets.Slider(period_ax, "Period", period_range[0], period_range[-1], valstep=period_range, valinit=initial_period)
    period_slider.on_changed(update)

    plt.show()


def cut(time, flux, period): 
    time_cut = []
    flux_cut = []
    cut = 0
    for idx in range(1, len(time)):
        if time[idx] % period < time[idx-1] % period:
            time_cut.append(time[cut:idx])
            flux_cut.append(flux[cut:idx])
            cut = idx

    time_cut = np.array(time_cut, dtype=object)
    flux_cut = np.array(flux_cut, dtype=object)
    return np.stack((time_cut, flux_cut), axis=-1)


def produceSingleTransitPlotsAnimation(transits_cut, phase, flux, period, write=False):
    fig, ax = plt.subplots()
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    fig.suptitle("Varying transits plotted over the complete folded light curve")

    ax.set_xlim(np.min(phase)-np.std(phase), np.max(phase) + np.std(phase))
    ax.set_ylim(np.min(flux)-np.std(flux), np.max(flux)+np.std(flux))
    folded, = ax.plot([], [], label="Complete folded light curve", lw=0, marker='o', markersize=0.5)
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

    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=transits_cut, interval=20, blit=True)

    if write:
        anim.save("TransitAnimation.mp4", writer)

    plt.show()


def produceSingleTransitPlotsInteractive(transits_cut, phase, flux, period):
    fig, ax = plt.subplots()
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    fig.suptitle("Varying transits plotted over the complete folded light curve")

    n_transits = len(transits_cut)

    transit_time, transit_flux = transits_cut[0]
    transit_phase = fold(transit_time, period)
    folded, = ax.plot(phase, flux, label="Complete folded light curve", lw=0, marker='o', markersize=0.5)
    transit, = ax.plot(transit_phase, transit_flux, color="r", label="Nth transit")
    minimum, = ax.plot(transit_phase[np.argmin(transit_flux)], np.min(transit_flux), label="Min", marker='o', markersize=4, color='b')
    maximum, = ax.plot(transit_phase[np.argmax(transit_flux)], np.max(transit_flux), label="Max", marker='o', markersize=4, color='g')
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
    transit_slider = matplotlib.widgets.Slider(transit_ax, "Number transit", 0, n_transits-1, valinit=0, valstep=range(n_transits))
    transit_slider.on_changed(update)

    plt.show()


def customBootstrap(x_data, y_data, n_samples):
    x_samples = np.zeros((n_samples, x_data.shape[0]))
    y_samples = np.zeros((n_samples, y_data.shape[0]))
    sample_idx = np.indices(y_data.shape)[0]

    for idx in range(n_samples):
        resample_idx = np.sort(np.random.choice(sample_idx, size=y_data.shape))
        x_samples[idx] = x_data[resample_idx]
        y_samples[idx] = y_data[resample_idx]
    
    return x_samples, y_samples


def customResidualBootstrap(data, n_samples, block_size, smooth_func):
    sample = np.pad(data, (0, block_size), mode="median")

    smooth_data = smooth_func(sample)
    residuals = sample - smooth_data
    samples = np.zeros((n_samples, sample.shape[0]))

    for n in range(n_samples):
        for idx in range(len(residuals)-block_size):
            residual_block = residuals[idx:idx + block_size]
            smooth_block = smooth_data[idx:idx + block_size]
            samples[n][idx:idx + block_size] = smooth_block + np.random.choice(residual_block, size=residual_block.shape)
    
    return samples[:, :-block_size]


if __name__ == "__main__":

    """
    Target List:
    target: Kepler-78b, notes: primary target for analysis
    target: Kepler-1520b, notes: disintegrating planet
    """

    # Initialize Target Constants
    target = "Kepler-78 b"
    if target in sp_csv.index:
        target_period_range = getPeriodRange(sp_csv.loc[target, "pl_orbper"], up_buffer=1)
        target_duration = sp_csv.loc[target, "pl_trandur"]/24
        if np.isnan(target_duration):
            target_duration = 1/24
    else:
        print("Target not found in provided CSV file")
        target_period_range = None
        target_duration = 1/24
    quarter = 2
    filter_cutoff = 0.5

    print("=" * 100)

    # Obtain data using lightkurve (Warning happens on following line)
    hdu = obtainFitsLightCurve(quarter=quarter, write_meta=False, target=target, mission="Kepler", exptime="long")
    lc = dataFromHDU(hdu, flux_column_name="FLUX")
    print(f"Quarter: {quarter}")

    # Preprocess data by removing Nan values, and filtering out long term trends
    num_nans = fillNans(lc.flux)
    produceTrendPlots(lc.time, lc.flux, filter_cutoff, lowPassGaussian)
    trend = filter(lc.flux, filter_cutoff, lowPassGaussian)
    # trend = getMedianLine(lc.time, lc.flux, 1, sorted=True)
    lc.flux = lc.flux/trend - 1.0
    lc.time, lc.flux, num_outliers = removeOutliers(lc.time, lc.flux, 8)
    print(f"{num_nans} nan flux values were filled in, {num_outliers} outliers were removed")
    print(f"Number of data points: {len(lc.time)}, Baseline: {lc.time[-1] - lc.time[0]} days")

    # Obtaining new data and constants with a variety of data analysis techniques
    period = getPeriod(lc.time, lc.flux, target_duration, period=target_period_range)
    spacing = 0.01 * (period ** 2 / (lc.time[-1] - lc.time[0]))
    period_grid = np.arange(period-spacing*100, period+spacing*99, spacing)
    lc.fold(period)
    transits_cut = cut(lc.time, lc.flux, period)
    print(f"Period of {target} obtained from BLS periodogram: {period} Days or {period * 24} Hours")

    print("=" * 100)

    # Producing a variety of informative plots and interactive plots
    produceBLSPeriodogramPlots(lc.time, lc.flux, kep78b_duration, period=target_period_range)
    produceFoldPlots(lc.time, lc.flux, period, time_bin_size=1)
    produceFoldPlotsInteractive(lc.time, lc.flux, period_grid, target_duration)
    produceSingleTransitPlotsInteractive(transits_cut, lc.phase, lc.flux, period)
    # produceFoldPlotsAnimation(lc.time, lc.flux, period_grid, target_duration)

    # Bootstrap Test
    # ===============================================================================================
    # Instantiating the figure and pre-processing the folded transit data
    samples = 100
    fig, (ax1) = plt.subplots(1, 1)
    fig.set(figheight=winSize[1], figwidth=winSize[0])
    ax1.set_title(f"Bootstrapped Folded Light Curve ({samples} samples)")  
    # ax2.set_title("Residuals Used for Bootstrapping")  
    lc.sortToPhase()
    smooth_func = lambda x: filter(x, 1.0, lowPassGaussian)

    # Custom tsmoothie smoother to override smoother restrictions
    class CustomSmoother(smoother._BaseSmoother):
        def __init__(self, smooth_func):
            super(CustomSmoother, self).__init__()
            self.smooth_func = smooth_func
         
        def smooth(self, data):
            _check_data(data)

            smooth = self.smooth_func(data)

            _check_output(smooth)
            _check_output(data)

            self._store_results(smooth_data=smooth, data=data)

            return self

    # Preforming residual resampling with tmoosthie functions 
    """
    bts_smoother = CustomSmoother(lambda x: np.expand_dims(smooth_func(x), axis=0))
    bts = bootstrap.BootstrappingWrapper(bts_smoother, bootstrap_type="mbb", block_length=10)
    samples = bts.sample(flux, n_samples=samples)
    """

    # Preforming residual resampling with a custom function
    """
    samples = customResidualBootstrap(flux, samples, 10, smooth_func)
    """

    # Preforming classic resampling with a custom function
    phase_samples, samples = customBootstrap(lc.phase, lc.flux, samples)

    # Plotting the original folded transit data and the bootstrap samples on top
    flux_binned = getMedianLine(lc.phase, lc.flux, 2, sorted=True)
    smooth_data = smooth_func(lc.flux)
    samples = np.apply_along_axis(lambda x: getMedianLine(lc.phase, x, 2, sorted=True), 1, samples)
    upper = np.apply_along_axis(lambda x: np.percentile(x, 95), 0, samples)
    lower = np.apply_along_axis(lambda x: np.percentile(x, 5), 0, samples)

    ax1.scatter(lc.phase, lc.flux, s=2)
    # ax1.fill_between(phase, lower, upper, alpha=0.8, color="orange")
    for p, f in zip(phase_samples, samples):
        ax1.plot(p, f, alpha=0.3, color="orange")
    ax1.plot(lc.phase, flux_binned, color="red")
    # ax1.plot(phase, smooth_data, color="green")
    # ax2.scatter(lc.phase, lc.flux-smooth_data, s=2)
    # ===============================================================================================

    plt.show()
    