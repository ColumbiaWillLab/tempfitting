from numpy import sum, power, array, pi, exp, subtract, divide, argmin, log, mean, linspace, round, absolute, sqrt
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, fourier_ellipsoid
from os import listdir
from os.path import isfile, join
from time import time


# constants of the universe
mu_0 = 4 * pi * 10.**-7
hbar = 1.0545718 * 10.**-34
c = 299792458
mu_b = hbar * 2 * pi * 1.39962460 * 10.**6
k_b = 1.38 * 10**-23

# sodium constants
Isat = 6.26 * 10
Gamma = 2 * pi * 9.7946 * 10.**6
f0 = 508.8487162 * 10.**12
k = 2 * pi * f0 / c
m = 22.989769 * 1.672623 * 10**-27

# Experiment constants
pixel = 0.00375

def sigma(v, sigma_0, t):
    return sigma_0 + v * t

def get_Directory(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def group_names(list_names):
    names = []
    for i in range(int(len(list_names)/3)):
        names.append([list_names[3*i], list_names[3*i+1], list_names[3*i+2]])
    return names

def transmission(directory, names):
    data, laser, dark = imread(directory + '/' + names[0]), imread(directory + '/' + names[1]), imread(directory + '/' + names[2])
    data, laser, dark = data[:,:,0].astype('float'), laser[:,:,0].astype('float'), dark[:,:,0].astype('float')
    laser = gaussian_filter(laser, sigma = 3)
    data = gaussian_filter(data, sigma = 3)
    atoms = subtract(data, dark)
    light = subtract(laser, dark)
    threshold = 7
    t = divide(atoms, light, where = light > threshold)
    t[light <= threshold] = 1
    t[t > 1] = 1
    return t

def find_center(image):
    x_project = sum(image, 0)
    y_project = sum(image, 1)
    return argmin(x_project), argmin(y_project)

def AOI_crop(image, center, widths):
    if widths[0] < 250:
        widths[0] = 250
    if widths[1] < 250:
        widths[1] = 250
    
    x1, x2, y1, y2 = center[0] - widths[0] / 2, center[0] + widths[0] / 2, center[1] - widths[1] / 2, center[1] + widths[1] / 2
    
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    x_max, y_max = image.shape
    if x2 > x_max:
        x2 = x_max
    if y2 > y_max:
        y2 = y_max
        
    return image[int(y1):int(y2),int(x1):int(x2)]

def AOI_integration(image, center, widths, detuning, mag, v = 'no'):
        
    cropped_image = gaussian_filter(AOI_crop(image, center, widths), sigma = 3)
    
    s1 = -sum(sum(log(cropped_image), 0), 0)
    sigma = ( 3 * (2 * pi / k)**2 / (2 * pi) ) / (1 + (2 * detuning * 2 * pi * 10.**6 / Gamma)**2)
    Area = (3.75 / mag * 10.**-6)**2
    
    if v == 'yes':
        if widths[0] < 250:
            widths[0] = 250
        if widths[1] < 250:
            widths[1] = 250
        
        x1, x2, y1, y2 = center[0] - widths[0] / 2, center[0] + widths[0] / 2, center[1] - widths[1] / 2, center[1] + widths[1] / 2
        plot_cropped_image(image, cropped_image, x1, x2, y1, y2)
    
    return s1 * 10.**-6 * Area / sigma 

def integration(image, detuning, mag):
        
    cropped_image = gaussian_filter(image, sigma = 3)
    
    s1 = -sum(sum(log(cropped_image), 0), 0)
    sigma = ( 3 * (2 * pi / k)**2 / (2 * pi) ) / (1 + (2 * detuning * 2 * pi * 10.**6 / Gamma)**2)
    Area = (3.75 / mag * 10.**-6)**2
    
    return s1 * 10.**-6 * Area / sigma 

def gaussian_x(x, A, sigma_0, h, x0):
    return A * exp( - power( (x - x0)/sigma_0 , 2) / 2 ) + h

def gaussian_no_h(x, A, sigma_0, x0):
    return A * exp( - power( (x - x0)/sigma_0 , 2) / 2 ) + 0.01

def fit_1D_gaussians(image, center, no_h = 'no'):
    image = -log(image)
    x_project = mean(image, 0)
    y_project = mean(image, 1)
    xs = list(range(len(x_project)))
    ys = list(range(len(y_project)))
    
    if no_h == 'yes':
        popt_x, pcov_x = curve_fit(gaussian_no_h, xs, x_project, p0 = [1, 100, center[0]])
        popt_y, pcov_y = curve_fit(gaussian_no_h, ys, y_project, p0 = [1, 100, center[1]])
        
        A_x, sigma_x, x0 = popt_x
        error_sigma_x = pcov_x[1,1]
        A_y, sigma_y, y0 = popt_y
        error_sigma_y = pcov_y[1,1]
        
        h_x, h_y = 0, 0
    else:
        popt_x, pcov_x = curve_fit(gaussian_x, xs, x_project, p0 = [1, 100, 0, center[0]], bounds = (0, [999999, 10000, 100, 10000]))
        popt_y, pcov_y = curve_fit(gaussian_x, ys, y_project, p0 = [1, 100, 0, center[1]], bounds = (0, [999999, 10000, 100, 10000]))
        
        A_x, sigma_x, h_x, x0 = popt_x
        error_sigma_x = pcov_x[1,1]
        A_y, sigma_y, h_y, y0 = popt_y
        error_sigma_y = pcov_y[1,1]
    
    return A_x, sigma_x, h_x, x0, error_sigma_x, A_y, sigma_y, h_y, y0, error_sigma_y

def plot_cropped_image(image, cropped_image, x1, x2, y1, y2):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(cropped_image)
    ax2.imshow(image)
    
    ax2.plot([x1,x1], [y1,y2])
    ax2.plot([x1,x2], [y1,y1])
    ax2.plot([x2,x2], [y1,y2])
    ax2.plot([x1,x2], [y2,y2])
    
    plt.show()    

def plot_1D_fits(atoms, center, A_x, sigma_x, h_x, x0_x, A_y, sigma_y, h_y, x0_y):
    atoms = -log(atoms)
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.set_size_inches(15, 15, forward=True)
    
    ax1.imshow(atoms)
    
    x_project = mean(atoms, 0)
    x = range(len(x_project))
    x_fit = gaussian_x(x, A_x, sigma_x, h_x, x0_x)
    ax2.plot(x, x_project)
    ax2.plot(x, x_fit)
    
    y_project = mean(atoms, 1)
    y = range(len(y_project))
    y_fit = gaussian_x(y, A_y, sigma_y, h_y, x0_y)
    ax3.plot(y, y_project)
    ax3.plot(y, y_fit)
    
    plt.show()

def get_temp(ts, sigmas, error_sigma):
    popt, pcov = curve_fit(sigma, ts, sigmas)
    return m / k_b * popt[1] * 10.**6, m / k_b * pcov[1,1] * 10.**6, popt[0], popt[1]

def fit_widths(mypath, mag, detuning, i_s = [-1], time = 10, v = 'no', AOI = [0, 0, 1291, 963]):
    names = get_Directory(mypath)
    names = group_names(names)
    x1, y1, x2, y2 = AOI
    Ts = []
    Ns = []
    for i in i_s:
        atoms = gaussian_filter(transmission(mypath, names[int(i)])[int(y1):int(y2),int(x1):int(x2)], sigma = 3)
        center = find_center(atoms)
        sigma_x, sigma_y, error_x, error_y, A_x, h_x, x0_x, A_y, h_y, x0_y = fit_1D_gaussians(atoms, center)
        
        if v == 'yes':
            plot_1D_fits(atoms, center, A_x, sigma_x, h_x, x0_x, A_y, sigma_y, h_y, x0_y)
            
        T = ((sigma_x + sigma_y) / 2 * pixel * mag / time)**2 * 0.5 * 1.67 * 23 * 100 / 1.38
        N = AOI_integration(atoms, center, [int(6 * sigma_x), int(6 * sigma_y)], detuning, mag, v = 'no')
        print( ((sigma_x) * pixel * mag / time)**2 * 0.5 * 1.67 * 23 * 100 / 1.38 )
        Ts.append(T)
        Ns.append(N)
    return Ns, Ts

def fit_progression(mypath, mag, detuning, times, v = 'no', offset = 0, AOI = [0, 0, 1291, 963]):
    sigma_xs = []
    sigma_ys = []
    error_xs = []
    error_ys = []
    atom_num = []
    
    x1, y1, x2, y2 = AOI
    
    names = get_Directory(mypath)
    if len(names) > 3 * len(times):
        names = names[int(len(names)- 3 * len(times) - 3 * len(times) * offset):int(len(names) - 3 * offset * len(times))]
        
    names = group_names(names)
    print (names)
    atoms = gaussian_filter(transmission(mypath, names[0])[int(y1):int(y2),int(x1):int(x2)], sigma = 0)
    center = find_center(atoms)
    
    for i in names:
        atoms = gaussian_filter(transmission(mypath, i)[int(y1):int(y2),int(x1):int(x2)], sigma = 0)
        A_x, sigma_x, h_x, x0, error_sigma_x, A_y, sigma_y, h_y, y0, error_sigma_y = fit_1D_gaussians(atoms, center, no_h = 'no')
        if v == 'yes':
            print(sigma_x, sigma_y, error_sigma_x, error_sigma_y, A_x, h_x, x0, A_y, h_y, y0)
            plot_1D_fits(atoms, center, A_x, sigma_x, h_x, x0, A_y, sigma_y, h_y, y0)
        atom_num.append( AOI_integration( atoms, center, [sigma_x * 6, sigma_y * 6], detuning, mag ) )
        sigma_xs.append(sigma_x * pixel / mag)
        sigma_ys.append(sigma_y * pixel / mag)
        error_xs.append(error_sigma_x * pixel / mag)
        error_ys.append(error_sigma_y * pixel / mag)
        
    sigma_xs, sigma_ys = power(array(sigma_xs), 2), power(array(sigma_ys), 2) 
    
    error_xs, error_ys = power(array(error_xs), 2), power(array(error_ys), 2) 
    
    t_x, error_t_x, sigma_0_x, v_x = get_temp(times, sigma_xs, error_xs)
    fits_x = sigma(v_x, sigma_0_x, times)
    
    t_y, error_t_y, sigma_0_y, v_y = get_temp(times, sigma_ys, error_ys)
    fits_y = sigma(v_y, sigma_0_y, times)
    
    print (sqrt(sigma_0_x), sqrt(sigma_0_y))
    
    if v == 'yes':
        plot_progression(times, sigma_xs, error_xs, fits_x, sigma_ys, error_ys, fits_y)
        print ((t_x + t_y) / 2, atom_num)
        
    return t_x, t_y, atom_num
#    
#    return t_x, error_t_x, fits_x, sigma_xs, error_xs, t_y, error_t_y, fits_y, sigma_ys, error_ys, atom_num[0]

def plot_progression(times, sigma_xs, error_xs, fits_x, sigma_ys, error_ys, fits_y):
    fig, (ax1, ax2) = plt.subplots(2)
    
    ax1.scatter(times, sigma_xs)
    ax1.errorbar(times, sigma_xs, yerr = error_xs, fmt = 'o')
    ax1.plot(times, fits_x)    
    
    ax2.scatter(times, sigma_ys)
    ax2.errorbar(times, sigma_ys, yerr = error_ys, fmt = 'o')
    ax2.plot(times, fits_y)
    
    plt.show()

def main():
    t_Start = time()
    mag = .2
    detuning = 1.
    repump_time = 0.5
    times = power(array([6,8,10,12]) + repump_time, 2)
    #times = power(array([11, 12, 13]) + repump_time, 2)
    #times = power(array([8,9,10,11,12,13,14,15,16,17,18]) + repump_time, 2)
    
    mypath = 'C:/Users/Columbia/Documents/Imaging/Raw Data/2019-10-23'
    #mypath = 'C:/Users/Columbia/Documents/Python Scripts/for cal'
    
    n = 1
    AOI = [560, 475, 900, 625]
    
    #Ns, Ts = fit_widths(mypath, mag, detuning, i_s = linspace(-n, -1, n), v = 'yes', time = 10.4)
    #print (round(array(Ns), 2).tolist())
    #print (round(array(Ts), 2).tolist())
    
    #offsets = list(range(23))
    #for i in offsets:    
    #    t_x, t_y, num = fit_progression(mypath, mag, detuning, times, v = 'no', offset = i)
    #    print (i, t_x, t_y, t_x / 2 + t_y / 2, num)
    
    t_x, t_y, num = fit_progression(mypath, mag, detuning, times, v = 'yes', offset = 0)
    plt.show()
    print ('temperature (uK)', 'Number (10^6)')
    print (t_x / 2 + t_y / 2, mean(num), num, t_x, t_y)
    
    #plot_progression(times, sigma_xs, error_xs, fits_x, sigma_ys, error_ys, fits_y)
    

    

if __name__ == "__main__":
    main()