import time
from pyrpl import Pyrpl
import numpy as np
import scipy
import matplotlib.pyplot as plt

"""
********************************************************************************************************************
                                                    UTILS
********************************************************************************************************************
"""
def lPrime(x, A, B, g, x0):  # derivative a Lorantian
    return -2 * A * (x - x0) / (((x - x0) ** 2 + g ** 2) ** 2) + B





"""
********************************************************************************************************************
                                            RED PITAYA CLASS CONTROLLER
********************************************************************************************************************
"""
class RedPitayaController(object):
    """
    ********************************************************************************************************************
                                                        CONSTRUCTOR
    ********************************************************************************************************************
    """
    def __init__(self, hostname: str, user: str = 'root', password: str = 'root', config: str = 'fermi',
                 gui: bool = False):
        try:
            p = Pyrpl(hostname=hostname, user=user, password=password, config=config, gui=gui)
            self.redpitaya = p.rp  # Access the RedPitaya object in charge of communicating with the board
        except Exception as e:
            print(e)

    """
    ********************************************************************************************************************
                                            SETTING AND RESETTING PARAMETERS
    ********************************************************************************************************************
    """
    def reset(self) -> None:
        # Turn off arbitrary signal generator channel 0
        self._setAsg0(output_direct='off', amp=0, offset=0)
        # Turn off arbitrary signal generator channel 1
        self._setAsg1(output_direct='off', amp=0, offset=0)
        # Turn off I+Q quadrature demodulation/modulation modules
        self.redpitaya.iq0.output = 'off'
        self._setIQ0(output_direct='off')
        # Turn off PID module 0
        self._setPid0(0, 0, 0, 0, 'off')
        # Turn off dac2
        self.setdac2(0)

    def _setAsg0(self, waveform: str = 'halframp', output_direct: str = 'out1', amp: float = 0.5,
                 offset: float = 0.5, freq: float = 1e2) -> None:
        self.redpitaya.asg0.setup(waveform=waveform, output_direct=output_direct, trigger_source='immediately',
                                  offset=offset, amplitude=amp, frequency=freq)

    def _setAsg1(self, waveform: str = 'halframp', output_direct: str = 'out2', amp: float = 0.8,
                 offset: float = 0.0, freq: float = 25e6) -> None:
        self.redpitaya.asg1.setup(waveform=waveform, output_direct=output_direct, trigger_source='immediately',
                                  offset=offset, amplitude=amp, frequency=freq)

    def _setIQ0(self, frequency: float = 25e6, bandwidth: list = [2e6, 2e6], gain: float = 0.5, phase: int = 0,
                acbandwidth: float = 5e6, amplitude: float = 1., input: str = 'in1', output_direct: str = 'out2',
                output_signal: str = 'quadrature', quadrature_factor: int = 1) -> None:
        self.redpitaya.iq0.setup(frequency=frequency, bandwidth=bandwidth, gain=gain, phase=phase,
                                 acbandwidth=acbandwidth, amplitude=amplitude, input=input, output_direct=output_direct,
                                 output_signal=output_signal, quadrature_factor=quadrature_factor)

    def setdac2(self, voltage: float = 0.) -> None:
        self.redpitaya.ams.dac2 = voltage   # pin 17 output 0

    def _setPid0(self, ival: float = 0, integrator: float = 1e3, proportional: float = 0,
                 differantiator: float = 0, input='iq0', output_direct: str = 'out1') -> None:
        # Clear integrator
        self.redpitaya.pid0.ival = ival
        # Proportinal
        self.redpitaya.pid0.p = proportional
        # Integrator
        self.redpitaya.pid0.i = integrator
        # differentiator
        self.redpitaya.pid0.d = differantiator
        # input or output
        self.redpitaya.pid0.input = input
        self.redpitaya.pid0.output_direct = output_direct

    """
    ********************************************************************************************************************
                                                    OSCILLOSCOPE
    ********************************************************************************************************************
    """
    def _scopeTrace(self) -> list:
        # set scope decimation factor (the scope module only has so much memory)
        self.redpitaya.scope.decimation = 256
        # The frequency corresponding to the length of one scope trace
        # freq = 1/(8E-9*(2**14)*scope.decimation)
        # Scope's first input
        self.redpitaya.scope.input1 = 'out1'
        # Scope's second input
        self.redpitaya.scope.input2 = 'iq0'
        # Trigger Threshold
        self.redpitaya.scope.threshold = self.redpitaya.asg0.offset + self.redpitaya.asg0.amplitude / 2
        # Trigger Hysteresis
        self.redpitaya.scope.hysteresis = 0.01
        # Trigger Source
        self.redpitaya.scope.trigger_source = 'ch1_positive_edge'
        # Trigger Time Delay
        self.redpitaya.scope.trigger_delay = 0
        # Take a Scope Trace
        return self.redpitaya.scope.single()

    def scope(self, input1='out1', input2='in2', hysteresis=0.01, trigger_source='immediately') -> list:    # TODO turn to private
        self.redpitaya.scope.decimation = 256
        self.redpitaya.scope.input1 = 'out1'
        # Scope's second input
        self.redpitaya.scope.input2 = 'in2'
        self.redpitaya.scope.threshold = self.redpitaya.asg0.offset + self.redpitaya.asg0.amplitude / 2
        # Trigger Hysteresis
        self.redpitaya.scope.hysteresis = 0.01
        # Trigger Source
        self.redpitaya.scope.trigger_source = 'asg0' # TODO: check what happen if I comment this line
        # Trigger Time Delay
        self.redpitaya.scope.trigger_delay = 0
        # Take a Scope Trace"""
        self.redpitaya.scope.trigger_source = 'immediately'
        # self.redpitaya.scope.trigger_mode = 'rolling_mode'
        return self.redpitaya.scope.single()

    def _orderedScopeTrace(self, ordered=False):   # TODO turn to private (with boolean) if the scope showed by ordered or not
        purple_signal, blue_signal = self.scope()
        # find wave peak
        purple_signal_peak_index = np.where(purple_signal == purple_signal.max())[0][0]
        first_position = purple_signal_peak_index + 1
        purple_signal = np.concatenate((purple_signal[first_position:], purple_signal[:first_position]))
        blue_signal = np.concatenate((blue_signal[first_position:], blue_signal[:first_position]))
        return purple_signal, blue_signal


    """
    ********************************************************************************************************************
                                                TEMPERATURE SCANNING
    ********************************************************************************************************************
    """
    def scan_temperature(self, epsilon=1000) -> bool:
        for i in np.arange(0, 0.3, 0.00025):
            # set temperature
            self.setdac2(i)
            # take scope
            _, blue_signal = self._orderedScopeTrace()
            half_scope_trace = int(blue_signal.shape[0]/2)

            blue_signal_peak_index = np.where(blue_signal == blue_signal.max())[0][0]
            if half_scope_trace - epsilon < blue_signal_peak_index < half_scope_trace + epsilon and \
                    blue_signal.max() > .95:
                print(i)
                return True
        return False


    """
    ********************************************************************************************************************
                                                    PIEZO HANDLING
    ********************************************************************************************************************
    """
    def scanPiezo(self, asg: bool = True, output_direct: str = 'out1', amp: float = 0.5,
                  offset: float = 0.5, freq: float = 1e2) -> None:
        if asg:
            self._setAsg0(waveform='halframp', output_direct=output_direct, amp=amp, offset=offset, freq=freq)
        else:
            self._setAsg1(waveform='halframp', output_direct=output_direct, amp=amp, offset=offset, freq=freq)

    def ramp_piezo(self, phase=15):
        self.reset()
        self.scanPiezo(freq=1 / (8E-9 * (2 ** 14) * 256))
        self._setIQ0(phase=phase)

    """
    ********************************************************************************************************************
                                                    PDH LOCK
    ********************************************************************************************************************
    """
    def lockCavity(self, phase=20):
        print("Scan Piezo")
        self.scanPiezo(freq=1 / (8E-9 * (2 ** 14) * 256))
        print("Run on Modulation")
        self._setIQ0(phase=phase)
        print("Take a scope trace")
        scope_trace = self._scopeTrace()
        print("Done taking scope trace")
        np.save('scope_trace.npy', scope_trace)
        print("Saved scope trace")
        ch1, ch2 = scope_trace

        # Guess Initial Parameters
        print("Curve fit")
        offs = np.mean(ch2)
        gamma = (np.max(ch1) - np.min(ch1)) / 10
        x0 = (np.max(ch1) - np.min(ch1)) / 2
        amp = (np.max(ch2) - np.mean(ch2)) * (x0 ** 3)  # guessed,but don't guess the correct sign

        # Curve Fit
        poptLine, pcovLine = scipy.optimize.curve_fit(lPrime, ch1, ch2, p0=[amp, offs, gamma, x0])
        fit = lPrime(ch1, poptLine[0], poptLine[1], poptLine[2], poptLine[3])
        print("Plot")
        # Plot Measured Data and Curve Fit
        plt.figure(1)
        plt.title("PDH Error Signal")
        plt.xlabel("PZT Drive Voltage (V)")
        plt.ylabel("Error Signal (V)")
        plt.grid(True)
        plt.plot(ch1, ch2)
        plt.plot(ch1, fit)

        print("Go back to resonance")
        # Go to resonance (CONSTANT PIEZO)
        # self.constantPzt(V=poptLine[3])
        self._setAsg0(waveform='dc', output_direct='out1', offset=poptLine[3])

        print("Close the feedback loop")
        # Close the Feedback Loop
        # Set PID gains and corner frequencies
        # Set Point
        self.redpitaya.pid0.setpoint = poptLine[1]
        self._setPid0()

    def auto_relock(self):
        self.reset()
        self.ramp_piezo()
        if self.scan_temperature(1000):
            time.sleep(10)
            self.lockCavity()
            starting_time = time.time()
            print(f'Locked at: {starting_time}')
            while True:
                time.sleep(10)
                print(f'Seconds after start: {time.time() - starting_time}')
                #purple_signal, blue_signal = self._orderedScopeTrace()
                purple_signal, blue_signal = self.scope() # added
                print(f'purple (fast) signal mean: {purple_signal.mean()}')
                print(f'blue signal mean: {blue_signal.mean()}')
                if blue_signal.max() < 0.95:
                    end_time = time.time()
                    print(f'Lost lock at: {end_time}. Took {end_time-starting_time}')
                    break
        self.auto_relock()

    def psd_analysis_on_lock(self):
        self.reset()
        self.ramp_piezo()
        if self.scan_temperature(500):
            time.sleep(10)
            self.lockCavity()
            print('Cavity is locked, starting analysing every 1 second.')
            starting_time = time.time()
            print(f'Starting time: {starting_time}')
            while True:
                time.sleep(1)
                event_time = time.time()
                purple_signals = []
                for i in range(10):
                    purple_signal, blue_signal = self.scope()
                    purple_signals.append(purple_signal)
                    if blue_signal.max() < 0.95:
                        end_time = time.time()
                        print(f'Unlocked at time: {end_time}.'
                              f'\nTotal seconds elapsed {end_time-starting_time}.'
                              f'\nBlocked at {i}.')
                        self._psd_analysis_on_lock(purple_signals, starting_time, event_time)
                        break
                self._psd_analysis_on_lock(purple_signals, starting_time, event_time)

    def _psd_analysis_on_lock(self, purple_signals, starting_time, event_time):
        fs = 488281.25
        f_list, pxx_list = [], []
        #np.save(f'psd_analysis/{event_time}-fast_outputs.npy', purple_signals)
        print(f'Analysis after {int(event_time - starting_time)} seconds')
        for i, purple_signal in enumerate(purple_signals):
            print(f'Average voltage {i}: {purple_signal.mean()}')
            f, pxx = scipy.signal.welch(purple_signal, fs, nperseg=len(purple_signal)//2)
            f_list.append(f)
            pxx_list.append(pxx)
            i = np.argmax(pxx)
            print(f[i])
            print('loglog')
            plt.loglog(f, pxx)
            #plt.axvline(f[i], c='r', alpha=0.5)
            plt.show()
            plt.clf()
            print('semilogy')
            plt.semilogy(f, pxx)
            #plt.axvline(f[i], c='r', alpha=0.5)
            plt.show()
        print('avg')
        f = [sum(sub_list) / len(sub_list) for sub_list in zip(*f_list)]
        pxx = [sum(sub_list) / len(sub_list) for sub_list in zip(*pxx_list)]
        i = np.argmax(pxx)
        print(f[i])
        plt.loglog(f, pxx)
        #plt.axvline(f[i], c='r', alpha=0.5)
        plt.show()
        plt.clf()
        plt.semilogy(f, pxx)
        #plt.axvline(f[i], c='r', alpha=0.5)
        plt.show()

    def play_with_temperature(self, increment, increase=False, check_time=10):
        self.reset()
        self.ramp_piezo()
        if self.scan_temperature(500):
            time.sleep(10)
            self.lockCavity()
            print(f'Cavity is locked, starting analysing every {check_time} second.')
            starting_time = time.time()
            print(f'Locked at: {starting_time}')
            original_dac2 = self.redpitaya.ams.dac2
            while True:
                time.sleep(check_time)
                if increase:
                    self.setdac2(self.redpitaya.ams.dac2 + increment)
                else:
                    self.setdac2(original_dac2 + increment)
                print(f'Seconds after start: {time.time() - starting_time}')
                purple_signal, blue_signal = self.scope()
                print(f'purple (fast) signal mean: {purple_signal.mean()}')
                print(f'blue signal mean: {blue_signal.mean()}')
                if blue_signal.max() < 0.95:
                    end_time = time.time()
                    print(f'Unlocked at time: {end_time}.'
                          f'\nTotal seconds elapsed {end_time - starting_time}.')
                    #self._psd_analysis_on_lock(purple_signals, starting_time, event_time)
                    break
                #self._psd_analysis_on_lock(purple_signals, starting_time, event_time)


if __name__ == '__main__':
    rp = RedPitayaController('169.254.167.128')
