#include "parse.h"
#include <boost/spirit/include/qi.hpp>
#include <boost/algorithm/string.hpp>

static boost::spirit::qi::real_parser<double, boost::spirit::qi::strict_real_policies<double>> const strict_double;

int countChar(const char* s, char c)
{
    int n = 0;
    while(*s++ != '\0') { //Until the end of the string
        if(*s == c) {
            n++;
        }
    }
    return n;
}
char** splitZeroCopyInPlace(char* s, char delimiter, int* size)
{
    int numDelimiters = countChar(s, delimiter);
    //Allocate memory
    char** splitResult = (char**)malloc((numDelimiters + 1) * sizeof(char*));
    //First substring starts at the beginning of s
    splitResult[0] = s;
    //Find other substrings
    int i = 1;
    char* hit = s;
    while((hit = strchr(hit, delimiter)) != NULL) { //Find next delimiter
        //In-place replacement of the delimiter
        *hit++ = '\0';
        //Next substring starts right after the hit
        splitResult[i++] = hit;
    }
    *size = numDelimiters + 1;
    return splitResult;
}

double parse(std::string const& s)
{
    //std::istringstream iss(s);
    //double d;
    //iss >> d;
    //return d;
    typedef std::string::const_iterator It;
    It f(begin(s)), l(end(s));
    static boost::spirit::qi::rule<It, double> const p = strict_double;

    double a = 1.0;
    boost::spirit::qi::parse(f,l,p,a);

    return a;
}

