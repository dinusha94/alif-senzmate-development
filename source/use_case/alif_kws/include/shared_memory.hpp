#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <cstring>
#include <string>   // For std::string

// Define the shared buffer and synchronization flag in a shared memory region
// `volatile` ensures changes are visible across cores immediately
volatile char sharedBuffer[256];  // Shared buffer for strings
volatile bool dataReady = false;  // Flag to indicate data is ready

// Helper function to write data into shared buffer
inline void writeSharedBuffer(const char* message) {
    std::strcpy((char*)sharedBuffer, message);  // Copy the string to shared buffer
    dataReady = true;  // Set the flag indicating data is available
}

// Helper function to read data from shared buffer
inline std::string readSharedBuffer() {
    if (dataReady) {
        std::string result = std::string(const_cast<char*>(sharedBuffer)); 
        dataReady = false;  // Clear the flag
        return result;
    } else {
        return "";  // No data available
    }
}

#endif // SHARED_MEMORY_H
