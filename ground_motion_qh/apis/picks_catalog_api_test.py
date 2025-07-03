import os
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from obspy import UTCDateTime
import pandas as pd

from ground_motion_qh.apis import picks_catalog_api


class PicksCatalogMethodsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='no_trigger_foreacst_temporal', amp_diff=10, amp_shift=0,
           time_shift=0, mid_buffer=0.01),
      dict(testcase_name='w_trigger_foreacst_temporal', amp_diff=10, amp_shift=0,
           time_shift=0.1, mid_buffer=0.01),
      dict(testcase_name='w_trigger_foreacst_temporal_w_amp_shift', amp_diff=10, amp_shift=3,
           time_shift=0.1, mid_buffer=0.01),
  )
  def test_create_pairs(self, amp_diff, amp_shift, time_shift, mid_buffer):
    # Create a sample DataFrame

    # amp_diff = 10
    # amp_shift = 0
    # time_shift = 0
    # mid_buffer = 0.01
    df = self._mock_picks_catalog(
        cat_len=1_000, amp_diff=amp_diff, time_shift=time_shift, amp_shift=amp_shift)

    # Test the create_pairs method
    pairs_array, time_differences_sec = df.pick_catalog.create_pairs(
        trigger_column='amp_a',
        forecast_column='amp_b',
        trigger_threshold=-np.inf,
        sequential=False,
        trigger_time_column='p_epoch_time',
        forecast_time_column='amp_epoch_time',
        mid_buffer=mid_buffer,
        forecast_window_time=2
    )

    expected_time_diff = 1 if (mid_buffer > time_shift) else time_shift
    np.testing.assert_allclose(
        time_differences_sec, np.full(time_differences_sec.shape, expected_time_diff), rtol=1e-5
    )
    self.assertIsInstance(pairs_array, np.ndarray)
    self.assertTrue(all(np.diff(pairs_array, axis=1) == -
                    (amp_shift+(amp_diff if (mid_buffer > time_shift) else 0))))

  def _mock_picks_catalog(self, cat_len=1_000, amp_diff=10, time_shift=0.1,
                          amp_shift=0):
    amp_a = np.arange((cat_len*amp_diff), 0, step=-amp_diff)
    data = {
        # decreasing differenecs jumps of amp_diff
        'amp_a': amp_a,
        'amp_b': amp_a - np.abs(amp_shift),  # shifted by amp_shift
        'p_epoch_time': np.arange(cat_len),  # time diffs of 1sec
        # same as p_epoch_time but shifted
        'amp_epoch_time': np.arange(cat_len) + time_shift,
    }
    return pd.DataFrame(data)


if __name__ == '__main__':
  absltest.main()
