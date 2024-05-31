from enum import Enum

from mcculw import ul
from mcculw.enums import InterfaceType
from mcculw.enums import ScanOptions, FunctionType, Status
from mcculw.device_info import DaqDeviceInfo


class USB3101FS:

    VOLTAGE_RANGE = [-10,10] # pretty sure this is fixed and cant change
    
    
    class Units(Enum):
        BITS=0
        VOLTS=1


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

    def set_output(self, channel, value, units=Units.BITS):
        """
        Slowly sets a value one at a time. For repeated values, instead use set_scan_values.
        Can only set one channel at a time
        """

        ul.a_out(self.board_num, channel, self.ao_range, value)
        
    
    def set_scan_values(self,):
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
            raise Exception('Error: No DAQ devices found')

        print('Found', len(devices), 'DAQ device(s):')
        for device in devices:
            print('  ', device.product_name, ' (', device.unique_id, ') - ',
                'Device ID = ', device.product_id, sep='')

        device = devices[0]
        if dev_id_list:
            device = next((device for device in devices
                        if device.product_id in dev_id_list), None)
            if not device:
                err_str = 'Error: No DAQ device found in device ID list: '
                err_str += ','.join(str(dev_id) for dev_id in dev_id_list)
                raise Exception(err_str)

        # Add the first DAQ device to the UL with the specified board number
        ul.create_daq_device(board_num, device)


    ############### Unit things
    @staticmethod
    def bits_to_volts(bit_val):
        return 



if __name__ == "__main__":
    dev = USB3101FS.detect_auto()
    print("connected")