#include "ArduinoConfig.hpp"

#ifdef NO_FUNEXCEPT_DEFINITIONS
namespace std {
void __throw_bad_alloc() {
    Serial.println("bad_alloc");
    while (true) {
    }
}
void __throw_length_error(char const *c) {
    Serial.println("length_error");
    Serial.println(c);
}
} // namespace std
#endif