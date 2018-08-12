#ifndef _H_PARSE
#define _H_PARSE
#include <string>

/**
 * Count the number of occurrences of c in s
 */

int countChar(const char* s, char c);

/**
 * Split the string at delimiters using a zero-copy in-place algorithm
 * @param s The string to split (every delimiter will be replaced by NUL)
 * @param delimiter The delimiter character to split at
 * @param size The size of the return array w
 * @return A list of char* (size stored in *size), pointing to all split results in order
 */

char** splitZeroCopyInPlace(char* s, char delimiter, int* size);

double parse(std::string const& s);

#endif
