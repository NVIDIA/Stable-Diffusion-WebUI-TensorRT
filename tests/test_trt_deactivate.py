import unittest
from unittest.mock import patch, MagicMock
from scripts.trt import TrtUnet


class TestTrtUnetDeactivate(unittest.TestCase):
    def setUp(self):
        self.unet = TrtUnet("model", [{"filepath": "engine.plan"}])

    @patch("scripts.trt.Engine")
    def test_deactivate_allows_reactivation(self, mock_engine_cls):
        engine_instance = MagicMock()
        mock_engine_cls.return_value = engine_instance

        self.unet.activate()
        self.assertIsNotNone(self.unet.engine)

        self.unet.deactivate()
        self.assertIsNone(self.unet.engine)

        self.unet.activate()
        self.assertIs(self.unet.engine, engine_instance)
        engine_instance.load.assert_called()


if __name__ == "__main__":
