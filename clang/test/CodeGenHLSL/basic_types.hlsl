// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - -DNAMESPACED| FileCheck %s


// CHECK: @uint16_t_Val = external hidden addrspace(2) global i16, align 2
// CHECK: @int16_t_Val = external hidden addrspace(2) global i16, align 2
// CHECK: @uint_Val = external hidden addrspace(2) global i32, align 4
// CHECK: @uint64_t_Val = external hidden addrspace(2) global i64, align 8
// CHECK: @int64_t_Val = external hidden addrspace(2) global i64, align 8
// CHECK: @int16_t2_Val = external hidden addrspace(2) global <2 x i16>, align 4
// CHECK: @int16_t3_Val = external hidden addrspace(2) global <3 x i16>, align 8
// CHECK: @int16_t4_Val = external hidden addrspace(2) global <4 x i16>, align 8
// CHECK: @uint16_t2_Val = external hidden addrspace(2) global <2 x i16>, align 4
// CHECK: @uint16_t3_Val = external hidden addrspace(2) global <3 x i16>, align 8
// CHECK: @uint16_t4_Val = external hidden addrspace(2) global <4 x i16>, align 8
// CHECK: @int2_Val = external hidden addrspace(2) global <2 x i32>, align 8
// CHECK: @int3_Val = external hidden addrspace(2) global <3 x i32>, align 16
// CHECK: @int4_Val = external hidden addrspace(2) global <4 x i32>, align 16
// CHECK: @uint2_Val = external hidden addrspace(2) global <2 x i32>, align 8
// CHECK: @uint3_Val = external hidden addrspace(2) global <3 x i32>, align 16
// CHECK: @uint4_Val = external hidden addrspace(2) global <4 x i32>, align 16
// CHECK: @int64_t2_Val = external hidden addrspace(2) global <2 x i64>, align 16
// CHECK: @int64_t3_Val = external hidden addrspace(2) global <3 x i64>, align 32
// CHECK: @int64_t4_Val = external hidden addrspace(2) global <4 x i64>, align 32
// CHECK: @uint64_t2_Val = external hidden addrspace(2) global <2 x i64>, align 16
// CHECK: @uint64_t3_Val = external hidden addrspace(2) global <3 x i64>, align 32
// CHECK: @uint64_t4_Val = external hidden addrspace(2) global <4 x i64>, align 32
// CHECK: @half2_Val = external hidden addrspace(2) global <2 x half>, align 4
// CHECK: @half3_Val = external hidden addrspace(2) global <3 x half>, align 8
// CHECK: @half4_Val = external hidden addrspace(2) global <4 x half>, align 8
// CHECK: @float2_Val = external hidden addrspace(2) global <2 x float>, align 8
// CHECK: @float3_Val = external hidden addrspace(2) global <3 x float>, align 16
// CHECK: @float4_Val = external hidden addrspace(2) global <4 x float>, align 16
// CHECK: @double2_Val = external hidden addrspace(2) global <2 x double>, align 16
// CHECK: @double3_Val = external hidden addrspace(2) global <3 x double>, align 32
// CHECK: @double4_Val = external hidden addrspace(2) global <4 x double>, align 32

#ifdef NAMESPACED
#define TYPE_DECL(T)  hlsl::T T##_Val
#else
#define TYPE_DECL(T)  T T##_Val
#endif

#ifdef __HLSL_ENABLE_16_BIT
TYPE_DECL(uint16_t);
TYPE_DECL(int16_t);
#endif

// unsigned 32-bit integer.
TYPE_DECL(uint);

// 64-bit integer.
TYPE_DECL(uint64_t);
TYPE_DECL(int64_t);

// built-in vector data types:

#ifdef __HLSL_ENABLE_16_BIT
TYPE_DECL(int16_t2   );
TYPE_DECL(int16_t3   );
TYPE_DECL(int16_t4   );
TYPE_DECL( uint16_t2 );
TYPE_DECL( uint16_t3 );
TYPE_DECL( uint16_t4 );
#endif

TYPE_DECL( int2  );
TYPE_DECL( int3  );
TYPE_DECL( int4  );
TYPE_DECL( uint2 );
TYPE_DECL( uint3 );
TYPE_DECL( uint4     );
TYPE_DECL( int64_t2  );
TYPE_DECL( int64_t3  );
TYPE_DECL( int64_t4  );
TYPE_DECL( uint64_t2 );
TYPE_DECL( uint64_t3 );
TYPE_DECL( uint64_t4 );

#ifdef __HLSL_ENABLE_16_BIT
TYPE_DECL(half2 );
TYPE_DECL(half3 );
TYPE_DECL(half4 );
#endif

TYPE_DECL( float2  );
TYPE_DECL( float3  );
TYPE_DECL( float4  );
TYPE_DECL( double2 );
TYPE_DECL( double3 );
TYPE_DECL( double4 );
