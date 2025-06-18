import os
from absl.testing import absltest
from absl.testing import parameterized
from obspy import UTCDateTime
from scripts.Neri import download_data


class ParsingToolsTest(parameterized.TestCase):

  def test_time_str_parts_to_utc_datetime(self):
    # Test with a valid input
    time_str = '2023, 10, 1, 0, 0, 0'
    time_obj = download_data._time_str_parts_to_utc_datetime(time_str)
    self.assertEqual(time_obj.year, 2023)
    self.assertEqual(time_obj.month, 10)
    self.assertEqual(time_obj.day, 1)
    self.assertEqual(time_obj.hour, 0)
    self.assertEqual(time_obj.minute, 0)
    self.assertEqual(time_obj.second, 0)

    time_str_with_ms = '2024, 3, 7, 15, 20, 55, 123000'
    time_obj_with_ms = download_data._time_str_parts_to_utc_datetime(time_str_with_ms)
    self.assertEqual(time_obj_with_ms.year, 2024)
    self.assertEqual(time_obj_with_ms.month, 3)
    self.assertEqual(time_obj_with_ms.day, 7)
    self.assertEqual(time_obj_with_ms.hour, 15)
    self.assertEqual(time_obj_with_ms.minute, 20)
    self.assertEqual(time_obj_with_ms.second, 55)
    self.assertEqual(time_obj_with_ms.microsecond, 123000)

  def test_parse_time_line_from_file(self):
    # Test with a valid line
    line = "(2023, 10, 1, 0, 0, 0)"
    time_obj = download_data._parse_time_line_from_file(line)
    self.assertEqual(time_obj.year, 2023)
    self.assertEqual(time_obj.month, 10)
    self.assertEqual(time_obj.day, 1)
    self.assertEqual(time_obj.hour, 0)
    self.assertEqual(time_obj.minute, 0)
    self.assertEqual(time_obj.second, 0)

  @parameterized.parameters(
      "((2023, 10, 1, 0), (2024, 3, 7, 15, 20, 55, ))",
      "[(2023, 10, 1, 0, 0, 0,), (2024, 3, 7, 15, 20, 55 )]",
      " (2023, 10, 1, ) , (2024, 3, 7, 15, 20, 55, 123000) ",
  )
  def test_parse_times_string(self, times_string):
    # Test with a valid string input
    expected_output = [
        UTCDateTime(2023, 10, 1, 0, 0, 0),
        UTCDateTime(2024, 3, 7, 15, 20, 55, 123000 if "123000" in times_string else 0),
    ]
    parsed_output = download_data._parse_times_string(times_string)
    for po in parsed_output:
      self.assertIsInstance(po, UTCDateTime)
    self.assertEqual(parsed_output, expected_output)

  def test_parse_file_with_start_times(self):
    # Test with a valid file
    # The file should contain lines with single times in the format:
    # (YYYY1, MM1, DD1, HH1, MM1, SS1)
    # (YYYY2, MM2, DD2, HH2, MM2, SS2)
    expected_output = [
        UTCDateTime(2023, 10, 1, 0, 0),
        UTCDateTime(2024, 3, 7, 15, 20, 55),
        UTCDateTime(2023, 10, 2),
        UTCDateTime(2024, 3, 8, 16, 21, 56),
        UTCDateTime(1990, 1, 3),
        UTCDateTime(1991, 4, 24, 16, 59, 40),
        UTCDateTime(1905, 11, 5, 12, 0, 3, 500000),
        UTCDateTime(1915),
    ]
    parsed_output = download_data._parse_pairs_file(  # Assuming _parse_pairs_file is now _parse_file_with_start_times or similar
        os.path.join(os.path.dirname(__file__), 'download_data_test_times_file.txt'))
    for po in parsed_output:
      self.assertIsInstance(po, UTCDateTime)
    self.assertEqual(parsed_output, expected_output)

  @parameterized.parameters([
      "(2023, 10, 1, 0, 0, 0), (2024, 3, 7, 15, 20, 55), (1990, 1, 3)",
      os.path.join(os.path.dirname(__file__), 'download_data_test_times_file.txt'),
  ])
  def test_parse_start_times(self, input_val):
    # Test with a valid input string or file
    expected_output_str = [
        UTCDateTime(2023, 10, 1, 0, 0),
        UTCDateTime(2024, 3, 7, 15, 20, 55),
        UTCDateTime(1990, 1, 3),
    ]
    expected_output_file = [
        UTCDateTime(2023, 10, 1, 0, 0),
        UTCDateTime(2024, 3, 7, 15, 20, 55),
        UTCDateTime(2023, 10, 2),
        UTCDateTime(2024, 3, 8, 16, 21, 56),
        UTCDateTime(1990, 1, 3),
        UTCDateTime(1991, 4, 24, 16, 59, 40),
        UTCDateTime(1905, 11, 5, 12, 0, 3, 500000),
        UTCDateTime(1915),
    ]
    expected_output = expected_output_str if isinstance(
        input_val, str) and not os.path.isfile(input_val) else expected_output_file

    parsed_output = download_data._parse_start_times(input_val)
    for i in range(len(parsed_output)):
      self.assertIsInstance(parsed_output[i], UTCDateTime)
    self.assertEqual(parsed_output, expected_output)


if __name__ == '__main__':
  absltest.main()
