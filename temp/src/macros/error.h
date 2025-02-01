#ifndef DEBUG
#define DEBUG 1
#endif

#ifndef MACROS_H
#define MACROS_H

#if DEBUG == 1
#define ASSERT_ERR(condition, message)                 \
    do                                                 \
    {                                                  \
        if (!(condition))                              \
        {                                              \
            printf("Assertion failed: %s\n", message); \
            assert(condition);                         \
            exit(1);                                   \
        }                                              \
    } while (0)
#else
#define ASSERT_ERR(condition, message) \
    do                                 \
    {                                  \
    } while (0)
#endif

#endif