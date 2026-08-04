#include <stdint.h>

void bhpy_store32(volatile uint32_t *address, uint32_t value)
{
    *address = value;
}

uint32_t bhpy_load32(const volatile uint32_t *address)
{
    return *address;
}
