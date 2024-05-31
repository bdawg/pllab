from enum import Enum

from mcculw import ul
from mcculw.enums import InterfaceType
from mcculw.enums import ScanOptions, FunctionType, Status
from mcculw.device_info import DaqDeviceInfo

import numpy as np


class USB3101FS:

    VOLTAGE_RANGE = [-10.0, 10.0]  # pretty sure this is fixed and cant change
    BIT_DEPTH = 16

    class Units(Enum):
        BITS = 0
        VOLTS = 1

    def __init__(self, board_num) -> None:
        self.board_num = board_num

        daq_dev_info = DaqDeviceInfo(board_num)
        self.ao_info = daq_dev_info.get_ao_info()
        self.ao_range = self.ao_info.supported_ranges[0]

    @staticmethod
    def detect_auto():
        """
        detects and displays all available devices and selects the first device listed
        """
        board_num = 0
        USB3101FS._config_first_detected_device(board_num, [])

        daq_dev_info = DaqDeviceInfo(board_num)
        assert daq_dev_info.supports_analog_output
        assert daq_dev_info.product_name == "USB-3101FS"

        return USB3101FS(0)

    def set_output(self, channel, value, units=None):
        """
        Slowly sets a value one at a time. For repeated values, instead use set_scan_values.
        Can only set one channel at a time
        """
        if units == USB3101FS.Units.VOLTS:
            value = self.volts_to_bits(value)

        ul.a_out(self.board_num, channel, self.ao_range, int(value))

    def set_scan_values(
        self,
    ):
        """
        Sets a series of values
        """

    @staticmethod
    def _config_first_detected_device(board_num, dev_id_list=None):
        """Adds the first available device to the UL.  If a types_list is specified,
        the first available device in the types list will be add to the UL.

        Parameters
        ----------
        board_num : int
            The board number to assign to the board when configuring the device.

        dev_id_list : list[int], optional
            A list of product IDs used to filter the results. Default is None.
            See UL documentation for device IDs.
        """
        ul.ignore_instacal()
        devices = ul.get_daq_device_inventory(InterfaceType.ANY)
        if not devices:
            raise Exception("Error: No DAQ devices found")

        print("Found", len(devices), "DAQ device(s):")
        for device in devices:
            print(
                "  ",
                device.product_name,
                " (",
                device.unique_id,
                ") - ",
                "Device ID = ",
                device.product_id,
                sep="",
            )

        device = devices[0]
        if dev_id_list:
            device = next(
                (device for device in devices if device.product_id in dev_id_list), None
            )
            if not device:
                err_str = "Error: No DAQ device found in device ID list: "
                err_str += ",".join(str(dev_id) for dev_id in dev_id_list)
                raise Exception(err_str)

        # Add the first DAQ device to the UL with the specified board number
        ul.create_daq_device(board_num, device)

    ############### Unit things
    @classmethod
    def bits_to_volts(cls, bit_val):
        """
        assumes a bit is an integer in the range 0, 2**BIT_DEPTH - 1
        """
        return (cls.VOLTAGE_RANGE[1] - cls.VOLTAGE_RANGE[0]) * bit_val / (
            2**cls.BIT_DEPTH - 1
        ) + cls.VOLTAGE_RANGE[0]

    @classmethod
    def volts_to_bits(cls, volt_val):
        return np.floor(
            (2**cls.BIT_DEPTH - 1)
            * (volt_val - cls.VOLTAGE_RANGE[0])
            / (cls.VOLTAGE_RANGE[1] - cls.VOLTAGE_RANGE[0])
        )


def test_unit_conversions():
    print(USB3101FS.bits_to_volts(0), USB3101FS.bits_to_volts(2**16 - 1))
    np.testing.assert_allclose(USB3101FS.bits_to_volts(0), -10.0)
    np.testing.assert_allclose(USB3101FS.bits_to_volts(2**16 - 1), 10.0)
    assert USB3101FS.bits_to_volts(2**15 - 1) < 0.0
    assert USB3101FS.bits_to_volts(2**15) > 0.0
    print(
        USB3101FS.volts_to_bits(-10),
        USB3101FS.volts_to_bits(0),
        USB3101FS.volts_to_bits(10),
    )
    np.testing.assert_allclose(USB3101FS.volts_to_bits(-10), 0)
    np.testing.assert_allclose(
        USB3101FS.volts_to_bits(0), 2**15 - 1
    )  # expected behaviour is to floor
    np.testing.assert_allclose(USB3101FS.volts_to_bits(10), 2**16 - 1)


class PlanetSimulator:
    """
    A planet simulator for a single point source planet.
    This class drives the USB-3101FS to simulate the planet.
    It can only move it slowly. It can drive the tilt, tip and attenuator channels.
    It also handles units so that the user of this class doesn't need to send voltages to the USB-3101FS.
    """

    def __init__(self) -> None:
        self._setup_connections()

        # describe the mapping between the outputs of the USB and the physical device
        self.channel_map = {
            "tip": 0,
            "tilt": 1,
            "attenuator": 2,
        }

    def _setup_connections(self):
        self.analogue_out = USB3101FS.detect_auto()

    def set_position(self, position):
        """
        Set the position of the planet by moving the piezo mirror

        Parameters
        ----------
        position : array
            [tip, tilt] in normalised units, where -1 is the minimum and 1 is the maximum
        """
        # use the array to set analogue outputs
        voltages = self._position_to_voltage(position)

        print(f"setting voltages {voltages}")

        # self.analogue_out.set_output(self.channel_map["tip"], voltages[0], USB3101FS.Units.VOLTS)
        # self.analogue_out.set_output(self.channel_map["tilt"], voltages[1], USB3101FS.Units.VOLTS)

    def _position_to_voltage(self, position):
        """
        Converts the position to voltages
        """
        # taken from lab notes
        tip_voltage_bounds = [1.0, 7.0]
        tilt_voltage_bounds = [-5.0, 3.0]

        tip_voltage = (tip_voltage_bounds[1] - tip_voltage_bounds[0]) * position[
            0
        ] / 2 + (tip_voltage_bounds[1] + tip_voltage_bounds[0]) / 2

        tilt_voltage = (tilt_voltage_bounds[1] - tilt_voltage_bounds[0]) * position[
            1
        ] / 2 + (tilt_voltage_bounds[1] + tilt_voltage_bounds[0]) / 2

        return [tip_voltage, tilt_voltage]

    def set_contrast(self, contrast):
        """
        Set the contrast of the planet, relative to the main source.

        Parameters
        ----------
        contrast : float
            The contrast of the planet, where 0 is the minimum and 1 is the maximum
        """

        voltage = self._contrast_to_voltage(contrast)

        print(f"setting voltage {voltage}")

        # self.analogue_out.set_output(self.channel_map["attenuator"], voltage, USB3101FS.Units.VOLTS)

    def _contrast_to_voltage(self, contrast):
        """
        Converts the contrast to voltages
        """
        voltage_range = [0.0, 5.0]

        voltage = (voltage_range[1] - voltage_range[0]) * contrast / 2 + (
            voltage_range[1] + voltage_range[0]
        ) / 2

        return voltage


if __name__ == "__main__":
    import time

    # dev = USB3101FS.detect_auto()
    # print("connected")

    # for i in np.linspace(*dev.VOLTAGE_RANGE, 21):
    #     print(f"setting voltage {i}V")
    #     dev.set_output(0, i, USB3101FS.Units.VOLTS)
    #     time.sleep(0.01)
    # test_unit_conversions()

    ps = PlanetSimulator()

    ps.set_position([0, 0])
    ps.set_position([-1, 1])
    ps.set_position([1, -1])
