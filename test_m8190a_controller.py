"""M8190A 控制器单元测试与硬件诊断套件.

用法:
    # 1. 纯 Mock 测试（无需硬件，验证代码逻辑）
    python -m unittest test_m8190a_controller.py -v

    # 2. 真实硬件诊断（需要 M8190A 在线）
    python -m unittest test_m8190a_controller.TestM8190AHardware -v

    # 3. 运行全部
    python -m unittest discover -v
"""
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import pyvisa

# 假设用户代码保存在 m8190a_controller.py 中
# 如果文件名不同，请修改此导入
from m8190a_controller import M8190AController, quick_download_to_awg


class TestM8190AControllerMock(unittest.TestCase):
    """Mock 测试：无需真实硬件，验证代码逻辑与 SCPI 命令序列."""

    def setUp(self):
        """每个测试前创建 Mock 对象."""
        self.mock_rm = MagicMock()
        self.mock_inst = MagicMock()
        self.mock_rm.open_resource.return_value = self.mock_inst

        # 模拟 *IDN? 和 :TEST:PON? 返回
        self.mock_inst.query.side_effect = [
            "Keysight Technologies,M8190A,MY50000001,1.2.3\n",  # *IDN?
            "0\n",  # :TEST:PON?  -> 0 表示通过
        ]

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_connect_success(self, mock_resource_manager):
        """测试成功连接并验证设备身份."""
        mock_resource_manager.return_value = self.mock_rm

        awg = M8190AController(visa_addr="TCPIP0::192.168.1.10::5025::SOCKET")
        awg.connect()

        # 验证 open_resource 被正确调用
        self.mock_rm.open_resource.assert_called_once_with(
            "TCPIP0::192.168.1.10::5025::SOCKET"
        )

        # 验证超时和终止符设置
        self.assertEqual(self.mock_inst.timeout, 30000)
        self.assertEqual(self.mock_inst.write_termination, "\n")
        self.assertEqual(self.mock_inst.read_termination, "\n")

        # 验证身份查询
        self.mock_inst.query.assert_any_call("*IDN?")
        self.mock_inst.query.assert_any_call(":TEST:PON?")

        awg.close()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_connect_query_failure(self, mock_resource_manager):
        """测试连接时查询失败应抛出异常."""
        self.mock_inst.query.side_effect = pyvisa.errors.VisaIOError(
            "VI_ERROR_TMO"
        )
        mock_resource_manager.return_value = self.mock_rm

        awg = M8190AController()
        with self.assertRaises(pyvisa.errors.VisaIOError):
            awg.connect()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_configure_commands(self, mock_resource_manager):
        """测试 configure() 发送的 SCPI 命令序列是否正确."""
        mock_resource_manager.return_value = self.mock_rm

        awg = M8190AController(
            sample_rate=12e9,
            vpp=0.5,
            output_route="DC"
        )
        awg.connect()
        awg.configure(channels=(1, 2))

        # 检查参考时钟命令
        self.mock_inst.write.assert_any_call(":ROSC:FREQ 1e7")
        self.mock_inst.write.assert_any_call(":ROSC:SOUR EXT")

        # 检查通道 1 配置
        self.mock_inst.write.assert_any_call(":FREQ:RAST 12000000000.0")
        self.mock_inst.write.assert_any_call(":TRACe1:DWIDth WSP")
        self.mock_inst.write.assert_any_call(":OUTP1:ROUT DC")
        self.mock_inst.write.assert_any_call(":DC1:VOLT:AMPL 0.5")
        self.mock_inst.write.assert_any_call(":DC1:FORM NRZ")
        self.mock_inst.write.assert_any_call(":OUTP1:NORM ON")
        self.mock_inst.write.assert_any_call(":OUTP1:COMP ON")
        self.mock_inst.write.assert_any_call(":TRAC1:SEL 1")

        # 检查通道 2 配置
        self.mock_inst.write.assert_any_call(":OUTP2:ROUT DC")
        self.mock_inst.write.assert_any_call(":DC2:VOLT:AMPL 0.5")

        # 检查 OPC 查询
        self.mock_inst.query.assert_any_call("*OPC?")

        awg.close()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_configure_invalid_route(self, mock_resource_manager):
        """测试非法输出路径应抛出 ValueError."""
        mock_resource_manager.return_value = self.mock_rm
        awg = M8190AController(output_route="INVALID")
        awg.connect()

        with self.assertRaises(ValueError) as ctx:
            awg.configure()
        self.assertIn("Unsupported M8190A output route", str(ctx.exception))
        awg.close()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_scale_to_int16(self, mock_resource_manager):
        """测试波形缩放函数."""
        mock_resource_manager.return_value = self.mock_rm
        awg = M8190AController()
        awg.connect()

        # 全幅正弦波
        data = np.sin(np.linspace(0, 2*np.pi, 100))
        scaled = awg._scale_to_int16(data)

        # 验证类型和范围
        self.assertEqual(scaled.dtype, np.int16)
        self.assertLessEqual(np.max(scaled), 32764)   # 8191 * 4 = 32764
        self.assertGreaterEqual(np.min(scaled), -32764)

        # 验证 0 映射到 0
        self.assertEqual(scaled[len(scaled)//2], 0)  # sin(pi) ≈ 0

        awg.close()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_scale_to_int16_clipping(self, mock_resource_manager):
        """测试超过 [-1,1] 的输入会被归一化."""
        mock_resource_manager.return_value = self.mock_rm
        awg = M8190AController()
        awg.connect()

        data = np.array([2.0, -2.0, 0.0])  # 超出范围
        scaled = awg._scale_to_int16(data)

        # 应该被归一化到 [-1,1] 再映射
        self.assertEqual(np.max(scaled), 32764)
        self.assertEqual(np.min(scaled), -32764)
        self.assertEqual(scaled[2], 0)

        awg.close()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_download_waveform_commands(self, mock_resource_manager):
        """测试波形下载的 SCPI 命令序列和二进制传输."""
        mock_resource_manager.return_value = self.mock_rm
        awg = M8190AController()
        awg.connect()

        # 生成 1000 点测试波形
        data = np.zeros(1000)
        data[0] = 1.0
        data[500] = -1.0

        awg.download_waveform(data, channel=1, segment=1, run=True)

        # 验证 abort 和 segment 定义
        self.mock_inst.write.assert_any_call(":ABORt 1")
        self.mock_inst.write.assert_any_call(":TRACe1:DELete 1")
        self.mock_inst.write.assert_any_call(":TRACe1:DEFine 1,1000")

        # 验证二进制写入被调用
        self.mock_inst.write_binary_values.assert_called_once()
        args, kwargs = self.mock_inst.write_binary_values.call_args

        # 检查参数
        self.assertIn(":TRACe1:DATA 1,0,", args[0])
        self.assertEqual(kwargs["datatype"], "h")
        self.assertEqual(kwargs["is_big_endian"], False)
        self.assertEqual(kwargs["header_fmt"], "ieee")

        # 验证播放命令
        self.mock_inst.write.assert_any_call(":TRACe1:SELect 1")
        self.mock_inst.write.assert_any_call(":FUNCtion1:MODE ARBitrary")
        self.mock_inst.write.assert_any_call(":OUTPut1:STATe ON")
        self.mock_inst.write.assert_any_call(":INIT:IMM 1")

        awg.close()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_download_waveform_large_chunking(self, mock_resource_manager):
        """测试大波形分块传输逻辑（chunk > 523200）."""
        mock_resource_manager.return_value = self.mock_rm
        awg = M8190AController()
        awg.connect()

        # 生成 600000 点波形，超过 chunk 大小
        data = np.random.randn(600000)
        awg.download_waveform(data, channel=1, segment=1, run=False)

        # 验证 write_binary_values 被调用两次（分块）
        self.assertEqual(self.mock_inst.write_binary_values.call_count, 2)

        # 检查第一块和第二块的偏移量
        calls = self.mock_inst.write_binary_values.call_args_list
        self.assertIn(":TRACe1:DATA 1,0,", calls[0][0][0])
        self.assertIn(":TRACe1:DATA 1,523200,", calls[1][0][0])

        awg.close()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_download_iq(self, mock_resource_manager):
        """测试 IQ 双通道下载."""
        mock_resource_manager.return_value = self.mock_rm
        awg = M8190AController()
        awg.connect()

        iq = np.exp(1j * np.linspace(0, 2*np.pi, 1000))
        awg.download_iq(iq, channel_i=1, channel_q=2, segment=1, run=True)

        # 验证两个通道都下载了波形
        self.assertEqual(self.mock_inst.write_binary_values.call_count, 2)

        # 验证 I 和 Q 通道都触发了 INIT
        self.mock_inst.write.assert_any_call(":INIT:IMM 1")
        self.mock_inst.write.assert_any_call(":INIT:IMM 2")

        awg.close()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_stop(self, mock_resource_manager):
        """测试停止命令."""
        mock_resource_manager.return_value = self.mock_rm
        awg = M8190AController()
        awg.connect()

        awg.stop(channels=(1, 2))
        self.mock_inst.write.assert_any_call(":ABORt 1")
        self.mock_inst.write.assert_any_call(":ABORt 2")

        awg.close()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_context_manager(self, mock_resource_manager):
        """测试 with 语句自动关闭."""
        mock_resource_manager.return_value = self.mock_rm

        with M8190AController() as awg:
            self.assertIsNotNone(awg.inst)
            self.mock_inst.query.assert_called()  # connect() 已执行

        # 退出 with 后应关闭
        self.mock_inst.close.assert_called_once()
        self.mock_rm.close.assert_called_once()

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_write_before_connect_raises(self, mock_resource_manager):
        """测试未连接时写操作应抛出 RuntimeError."""
        mock_resource_manager.return_value = self.mock_rm
        awg = M8190AController()

        with self.assertRaises(RuntimeError) as ctx:
            awg.write("*IDN?")
        self.assertIn("AWG not connected", str(ctx.exception))

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_send_preset(self, mock_resource_manager):
        """测试预设复位."""
        mock_resource_manager.return_value = self.mock_rm
        awg = M8190AController()
        awg.connect()
        awg.send_preset()

        self.mock_inst.write.assert_any_call("*RST")
        self.mock_inst.query.assert_any_call("*OPC?")

        awg.close()


class TestM8190AHardware(unittest.TestCase):
    """真实硬件诊断测试.

    运行前请确保：
    1. M8190A 已开机并联网
    2. 无其他软件（BenchVue, NI MAX 等）占用该 VISA 地址
    3. 修改下面的 VISA_ADDR 为实际地址
    """

    VISA_ADDR = "TCPIP0::192.168.1.10::5025::SOCKET"
    # VISA_ADDR = "TCPIP0::localhost::hislip0::INSTR"  # 如果通过 HISLIP

    @classmethod
    def setUpClass(cls):
        """尝试连接硬件，如果失败则跳过全部测试."""
        try:
            cls.awg = M8190AController(visa_addr=cls.VISA_ADDR)
            cls.awg.connect()
            print(f"\n[硬件连接成功] {cls.awg.query('*IDN?')}")
        except Exception as e:
            cls.awg = None
            print(f"\n[硬件连接失败] {e}")

    @classmethod
    def tearDownClass(cls):
        if cls.awg:
            cls.awg.close()

    def setUp(self):
        if self.awg is None:
            self.skipTest("M8190A 硬件未连接，跳过硬件测试")

    def test_01_idn(self):
        """诊断 1: 读取设备身份."""
        idn = self.awg.query("*IDN?")
        print(f"  *IDN? = {idn}")
        self.assertIn("M8190A", idn)

    def test_02_power_on_test(self):
        """诊断 2: 上电自检."""
        pon = self.awg.query(":TEST:PON?")
        print(f"  :TEST:PON? = {pon}")
        # 0 = 通过, 1 = 失败
        self.assertIn(pon, ["0", "1"])

    def test_03_opc(self):
        """诊断 3: 操作完成查询（验证命令解析正常）."""
        self.awg.write("*CLS")
        opc = self.awg.query("*OPC?")
        print(f"  *OPC? = {opc}")
        self.assertEqual(opc, "1")

    def test_04_error_queue(self):
        """诊断 4: 检查错误队列是否为空."""
        err = self.awg.query("SYST:ERR?")
        print(f"  SYST:ERR? = {err}")
        # 期望 "+0,No error" 或类似
        self.assertIn("+0", err)

    def test_05_configure(self):
        """诊断 5: 配置 AWG 基本参数."""
        self.awg.reset()
        self.awg.configure(
            sample_rate=10e9,
            vpp=0.5,
            channels=(1,),
            output_route="DC"
        )
        # 如果配置成功，错误队列应为空
        err = self.awg.query("SYST:ERR?")
        self.assertIn("+0", err)

    def test_06_download_small_waveform(self):
        """诊断 6: 下载小波形并播放（核心功能验证）."""
        self.awg.reset()
        self.awg.configure(channels=(1,))

        # 生成 1024 点简单波形
        data = np.sin(2 * np.pi * np.arange(1024) / 1024)

        try:
            self.awg.download_waveform(data, channel=1, segment=1, run=True)
            err = self.awg.query("SYST:ERR?")
            print(f"  下载后 SYST:ERR? = {err}")
            self.assertIn("+0", err)
        except Exception as e:
            self.fail(f"波形下载失败: {e}")
        finally:
            self.awg.stop(channels=(1,))

    def test_07_download_large_waveform(self):
        """诊断 7: 下载大波形（触发分块传输）."""
        self.awg.reset()
        self.awg.configure(channels=(1,))

        # 600000 点，超过 523200 的 chunk 大小
        data = np.random.randn(600000)

        try:
            self.awg.download_waveform(data, channel=1, segment=1, run=False)
            err = self.awg.query("SYST:ERR?")
            print(f"  大波形下载后 SYST:ERR? = {err}")
            self.assertIn("+0", err)
        except Exception as e:
            self.fail(f"大波形下载失败: {e}")

    def test_08_download_iq(self):
        """诊断 8: IQ 双通道下载."""
        self.awg.reset()
        self.awg.configure(channels=(1, 2))

        iq = np.exp(1j * 2 * np.pi * np.arange(2048) / 2048)

        try:
            self.awg.download_iq(iq, channel_i=1, channel_q=2, segment=1, run=True)
            err = self.awg.query("SYST:ERR?")
            print(f"  IQ 下载后 SYST:ERR? = {err}")
            self.assertIn("+0", err)
        except Exception as e:
            self.fail(f"IQ 下载失败: {e}")
        finally:
            self.awg.stop(channels=(1, 2))


class TestQuickDownload(unittest.TestCase):
    """测试便捷函数 quick_download_to_awg."""

    @patch("m8190a_controller.pyvisa.ResourceManager")
    def test_quick_download(self, mock_resource_manager):
        """测试便捷函数调用流程."""
        mock_rm = MagicMock()
        mock_inst = MagicMock()
        mock_rm.open_resource.return_value = mock_inst
        mock_resource_manager.return_value = mock_rm

        data = np.ones(1000)

        # 需要模拟 config 模块中的默认值
        with patch("m8190a_controller.config") as mock_config:
            mock_config.AWG_SAMPLE_RATE = 12e9
            mock_config.AWG_VPP = 0.5
            mock_config.AWG_OUTPUT_ROUTE = "DC"
            mock_config.M8190A_VISA_ADDR = "TCPIP0::192.168.1.10::5025::SOCKET"

            quick_download_to_awg(data, channel=1)

        # 验证连接、配置、下载、关闭流程都被执行
        mock_rm.open_resource.assert_called_once()
        mock_inst.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
