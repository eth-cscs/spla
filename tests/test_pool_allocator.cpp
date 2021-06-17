#include <memory>
#include <cstddef>
#include <unordered_map>
#include <cstdlib>
#include <vector>
#include <deque>
#include <random>


#include "gtest/gtest.h"
#include "memory/allocator.hpp"
#include "memory/pool_allocator.hpp"

using namespace spla;


class PoolAllocatorTest : public ::testing::Test {
 protected:
   PoolAllocatorTest() {
     pool_.reset(new PoolAllocator<MemLoc::Host>(
         [&](std::size_t size) -> void* {
           void* ptr = std::malloc(size);
           bool newPointer = allocated_.emplace(ptr, size).second;
           // Pointer must not be allocated already(if free was called on pointer outside of
           // provided deallocate function, std::malloc might return a previous pointer again).
           EXPECT_TRUE(newPointer);
           return ptr;
         },
         [&](void* ptr) -> void {
           std::free(ptr);
           // make sure pointer was allocated and not freed yet
           EXPECT_TRUE(allocated_.erase(ptr));
         }));
   }

   void* call_allocate(std::size_t size) {
     void* ptr = pool_->allocate(size);

     auto it = allocated_.find(ptr);
     // Make sure returned pointer was actually allocated and not freed
     EXPECT_TRUE(it != allocated_.end());
     if(it != allocated_.end()) {
       // size must be within memory block size
       EXPECT_TRUE(it->second >= size);
     }

     return ptr;
   }

   void TearDown() override {
     pool_.reset();                    // Destroy pool
     EXPECT_TRUE(allocated_.empty());  // Make sure there is no memory leak
  }

  std::unordered_map<void*, std::size_t> allocated_;
  std::unique_ptr<Allocator<MemLoc::Host>> pool_;
};


TEST_F(PoolAllocatorTest, Serial) {
  std::vector<std::size_t> sizes = {1, 2, 2, 3, 4, 5, 4, 3};
  std::vector<void*> pointers;
  pointers.reserve(sizes.size());
  for(auto& s : sizes) {
    pointers.push_back(call_allocate(s));
  }
  for(auto& ptr : pointers) {
    pool_->deallocate(ptr);
  }
}

TEST_F(PoolAllocatorTest, Interleaved) {
  std::vector<std::size_t> sizes = {1, 2, 2, 3, 4, 5, 4, 3};
  for(auto& s : sizes) {
    auto ptr = call_allocate(s);
    pool_->deallocate(ptr);
  }
}

TEST_F(PoolAllocatorTest, Random) {
  const int numIter = 100 * 1000;
  std::mt19937 randGen;
  std::discrete_distribution<int> binaryDist(30, 70); // Allocate with 70% prob., deallocate with 30% prob.
  std::uniform_int_distribution<int> sizeDist(1, 100);

  std::deque<void*> pointers;

  for(int i = 0; i < numIter; ++i) {
    if(binaryDist(randGen)) {
      // Allocate
      pointers.emplace_back(call_allocate(sizeDist(randGen)));
    } else if(!pointers.empty()){
      // Deallocate
      std::uniform_int_distribution<int> indexDist(0, pointers.size() - 1);
      auto index = indexDist(randGen);
      auto it = pointers.begin() + index;
      pool_->deallocate(*it);
      pointers.erase(it);
    }
  }

  // deallocate remainder
  for(auto& ptr : pointers) {
    pool_->deallocate(ptr);
  }
}
