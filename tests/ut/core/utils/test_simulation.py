# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for the device-runtime simulation switch and stand-ins."""
import unittest

from hyper_parallel.core.utils.simulation import (
    SimulatedEvent,
    SimulatedStream,
    is_device_runtime_simulated,
    simulate_device_runtime,
    simulated_device_handle,
)


class TestSimulationSwitch(unittest.TestCase):
    """The process-wide switch toggles and reads back."""

    def tearDown(self):
        simulate_device_runtime(False)

    def test_switch_round_trip(self):
        """Enabling and disabling the switch is observable."""
        self.assertFalse(is_device_runtime_simulated())
        simulate_device_runtime(True)
        self.assertTrue(is_device_runtime_simulated())
        simulate_device_runtime(False)
        self.assertFalse(is_device_runtime_simulated())


class TestSimulatedPrimitives(unittest.TestCase):
    """Stand-ins answer the full stream/event protocol without a device."""

    def test_event_protocol(self):
        """Record/wait/query/synchronize complete immediately."""
        event = SimulatedEvent()
        self.assertIs(event.record(), event)
        event.record(SimulatedStream())
        event.wait()
        event.wait(SimulatedStream())
        self.assertTrue(event.query())
        event.synchronize()

    def test_stream_protocol(self):
        """Streams drain immediately and mint elapsed events."""
        stream = SimulatedStream()
        stream.synchronize()
        self.assertTrue(stream.query())
        stream.wait_event(SimulatedEvent())
        stream.wait_stream(SimulatedStream())
        self.assertTrue(stream.record_event().query())

    def test_device_handle_namespace(self):
        """The handle mirrors the torch.<device> surface swap code touches."""
        handle = simulated_device_handle()
        self.assertIsInstance(handle.Stream(), SimulatedStream)
        self.assertIsInstance(handle.Event(), SimulatedEvent)
        self.assertIs(handle.current_stream(), handle.current_stream())
        with handle.stream(handle.Stream()):
            pass
        handle.synchronize()
        handle.synchronize(0)
        with self.assertRaises(AttributeError):
            handle.get_device_name()  # pylint: disable=no-member


if __name__ == "__main__":
    unittest.main()
