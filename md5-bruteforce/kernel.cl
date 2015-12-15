#include "md5.cl"

__kernel void vector_add(__global const int *starts, __global const int *stops, __global const int *maxlen_in, __global uint *pw_hash, __global char *cracked_pw) {
  // Get the index of the current element to be processed
  __private int id = get_global_id(0);
  int start = starts[id];
  int stop = stops[id];
  int maxlen = *maxlen_in;
  char chars[32];

  for (int i = start; i < stop; i++) {
    int actual_length = 0;
    int next = i + 1;
    for (int j = 0; j <= maxlen; j++) {
      char c = (char) (97 + (next - 1) % 26);
      chars[j] = c;
      next = (next - 1) / 26;
      if (next == 0) {
        actual_length = j + 1;
        break;
      }
    }
    uint this_hash[4];
    md5(chars, actual_length, &this_hash);
    if (
      this_hash[0] == pw_hash[0] &&
      this_hash[1] == pw_hash[1] &&
      this_hash[2] == pw_hash[2] &&
      this_hash[3] == pw_hash[3]
      ) {
        printf("Found hash!\n");
        for (int i = 0; i < actual_length; i++) {
          cracked_pw[i] = chars[i];
          cracked_pw[i + 1] = 0x00;
        }
      }
  }
}
