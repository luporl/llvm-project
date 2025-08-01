; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefixes=GFX9 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 < %s | FileCheck -check-prefixes=GFX10 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck -check-prefixes=GFX11 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck -check-prefixes=GFX12 %s

define amdgpu_ps void @store_f16_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <2 x i32> %val) {
; GFX9-LABEL: store_f16_1d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x1 unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_f16_1d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_f16_1d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_f16_1d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.1d.v4f16.i16(<4 x half> %bitcast, i32 1, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_v2f16_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <2 x i32> %val) {
; GFX9-LABEL: store_v2f16_1d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x3 unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_v2f16_1d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_1D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_v2f16_1d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_1D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_v2f16_1d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_1D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.1d.v4f16.i16(<4 x half> %bitcast, i32 3, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_v3f16_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <2 x i32> %val) {
; GFX9-LABEL: store_v3f16_1d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x7 unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_v3f16_1d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_1D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_v3f16_1d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_1D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_v3f16_1d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_1D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.1d.v4f16.i16(<4 x half> %bitcast, i32 7, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_v4f16_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <2 x i32> %val) {
; GFX9-LABEL: store_v4f16_1d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0xf unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_v4f16_1d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_v4f16_1d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_v4f16_1d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.1d.v4f16.i16(<4 x half> %bitcast, i32 15, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_f16_2d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <2 x i32> %val) {
; GFX9-LABEL: store_f16_2d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x1 unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_f16_2d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_f16_2d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_f16_2d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.2d.v4f16.i16(<4 x half> %bitcast, i32 1, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_v2f16_2d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <2 x i32> %val) {
; GFX9-LABEL: store_v2f16_2d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x3 unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_v2f16_2d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_v2f16_2d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_v2f16_2d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.2d.v4f16.i16(<4 x half> %bitcast, i32 3, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_v3f16_2d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <2 x i32> %val) {
; GFX9-LABEL: store_v3f16_2d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x7 unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_v3f16_2d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_v3f16_2d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_v3f16_2d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_2D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.2d.v4f16.i16(<4 x half> %bitcast, i32 7, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_v4f16_2d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <2 x i32> %val) {
; GFX9-LABEL: store_v4f16_2d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0xf unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_v4f16_2d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_v4f16_2d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_v4f16_2d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[1:2], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.2d.v4f16.i16(<4 x half> %bitcast, i32 15, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_f16_3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi, <2 x i32> %val) {
; GFX9-LABEL: store_f16_3d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0x1 unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_f16_3d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_f16_3d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_f16_3d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[2:3], [v0, v1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.3d.v4f16.i16(<4 x half> %bitcast, i32 1, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_v2f16_3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi, <2 x i32> %val) {
; GFX9-LABEL: store_v2f16_3d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0x3 unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_v2f16_3d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_v2f16_3d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_v2f16_3d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[2:3], [v0, v1], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_3D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.3d.v4f16.i16(<4 x half> %bitcast, i32 3, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_v3f16_3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi, <2 x i32> %val) {
; GFX9-LABEL: store_v3f16_3d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0x7 unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_v3f16_3d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_3D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_v3f16_3d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_3D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_v3f16_3d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[2:3], [v0, v1], s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_3D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.3d.v4f16.i16(<4 x half> %bitcast, i32 7, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @store_v4f16_3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi, <2 x i32> %val) {
; GFX9-LABEL: store_v4f16_3d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0xf unorm a16 d16
; GFX9-NEXT:    s_endpgm
;
; GFX10-LABEL: store_v4f16_3d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D unorm a16 d16
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: store_v4f16_3d:
; GFX11:       ; %bb.0: ; %main_body
; GFX11-NEXT:    image_store v[2:3], v[0:1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D unorm a16 d16
; GFX11-NEXT:    s_endpgm
;
; GFX12-LABEL: store_v4f16_3d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    image_store v[2:3], [v0, v1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D a16 d16
; GFX12-NEXT:    s_endpgm
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  %bitcast = bitcast <2 x i32> %val to <4 x half>
  call void @llvm.amdgcn.image.store.3d.v4f16.i16(<4 x half> %bitcast, i32 15, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.image.store.1d.v4f16.i16(<4 x half>, i32, i16, <8 x i32>, i32, i32) #2
declare void @llvm.amdgcn.image.store.2d.v4f16.i16(<4 x half>, i32, i16, i16, <8 x i32>, i32, i32) #2
declare void @llvm.amdgcn.image.store.3d.v4f16.i16(<4 x half>, i32, i16, i16, i16, <8 x i32>, i32, i32) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
