//===- VectorTransferOpTransforms.cpp - transfer op transforms ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions concerned with optimizing transfer_read and
// transfer_write ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vector-transfer-opt"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;

/// Return the ancestor op in the region or nullptr if the region is not
/// an ancestor of the op.
static Operation *findAncestorOpInRegion(Region *region, Operation *op) {
  for (; op != nullptr && op->getParentRegion() != region;
       op = op->getParentOp())
    ;
  return op;
}

namespace {

class TransferOptimization {
public:
  TransferOptimization(RewriterBase &rewriter, Operation *op)
      : rewriter(rewriter), dominators(op), postDominators(op) {}
  void deadStoreOp(vector::TransferWriteOp);
  void storeToLoadForwarding(vector::TransferReadOp);
  void removeDeadOp() {
    for (Operation *op : opToErase)
      rewriter.eraseOp(op);
    opToErase.clear();
  }

private:
  RewriterBase &rewriter;
  bool isReachable(Operation *start, Operation *dest);
  DominanceInfo dominators;
  PostDominanceInfo postDominators;
  std::vector<Operation *> opToErase;
};

} // namespace
/// Return true if there is a path from start operation to dest operation,
/// otherwise return false. The operations have to be in the same region.
bool TransferOptimization::isReachable(Operation *start, Operation *dest) {
  assert(start->getParentRegion() == dest->getParentRegion() &&
         "This function only works for ops i the same region");
  // Simple case where the start op dominate the destination.
  if (dominators.dominates(start, dest))
    return true;
  return start->getBlock()->isReachable(dest->getBlock());
}

/// For transfer_write to overwrite fully another transfer_write must:
/// 1. Access the same memref with the same indices and vector type.
/// 2. Post-dominate the other transfer_write operation.
/// If several candidates are available, one must be post-dominated by all the
/// others since they are all post-dominating the same transfer_write. We only
/// consider the transfer_write post-dominated by all the other candidates as
/// this will be the first transfer_write executed after the potentially dead
/// transfer_write.
/// If we found such an overwriting transfer_write we know that the original
/// transfer_write is dead if all reads that can be reached from the potentially
/// dead transfer_write are dominated by the overwriting transfer_write.
void TransferOptimization::deadStoreOp(vector::TransferWriteOp write) {
  LLVM_DEBUG(DBGS() << "Candidate for dead store: " << *write.getOperation()
                    << "\n");
  llvm::SmallVector<Operation *, 8> blockingAccesses;
  Operation *firstOverwriteCandidate = nullptr;
  Value source = memref::skipViewLikeOps(cast<MemrefValue>(write.getBase()));
  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    // If the user has already been processed skip.
    if (!processed.insert(user).second)
      continue;
    if (isa<ViewLikeOpInterface>(user)) {
      users.append(user->getUsers().begin(), user->getUsers().end());
      continue;
    }
    if (isMemoryEffectFree(user))
      continue;
    if (user == write.getOperation())
      continue;
    if (auto nextWrite = dyn_cast<vector::TransferWriteOp>(user)) {
      // Check candidate that can override the store.
      if (memref::isSameViewOrTrivialAlias(
              cast<MemrefValue>(nextWrite.getBase()),
              cast<MemrefValue>(write.getBase())) &&
          checkSameValueWAW(nextWrite, write) &&
          postDominators.postDominates(nextWrite, write)) {
        if (firstOverwriteCandidate == nullptr ||
            postDominators.postDominates(firstOverwriteCandidate, nextWrite))
          firstOverwriteCandidate = nextWrite;
        else
          assert(
              postDominators.postDominates(nextWrite, firstOverwriteCandidate));
        continue;
      }
    }
    if (auto transferOp = dyn_cast<VectorTransferOpInterface>(user)) {
      // Don't need to consider disjoint accesses.
      if (vector::isDisjointTransferSet(
              cast<VectorTransferOpInterface>(write.getOperation()),
              cast<VectorTransferOpInterface>(transferOp.getOperation()),
              /*testDynamicValueUsingBounds=*/true))
        continue;
    }
    blockingAccesses.push_back(user);
  }
  if (firstOverwriteCandidate == nullptr)
    return;
  Region *topRegion = firstOverwriteCandidate->getParentRegion();
  Operation *writeAncestor = findAncestorOpInRegion(topRegion, write);
  assert(writeAncestor &&
         "write op should be recursively part of the top region");

  for (Operation *access : blockingAccesses) {
    Operation *accessAncestor = findAncestorOpInRegion(topRegion, access);
    // TODO: if the access and write have the same ancestor we could recurse in
    // the region to know if the access is reachable with more precision.
    if (accessAncestor == nullptr ||
        !isReachable(writeAncestor, accessAncestor))
      continue;
    if (!dominators.dominates(firstOverwriteCandidate, accessAncestor)) {
      LLVM_DEBUG(DBGS() << "Store may not be dead due to op: "
                        << *accessAncestor << "\n");
      return;
    }
  }
  LLVM_DEBUG(DBGS() << "Found dead store: " << *write.getOperation()
                    << " overwritten by: " << *firstOverwriteCandidate << "\n");
  opToErase.push_back(write.getOperation());
}

/// A transfer_write candidate to storeToLoad forwarding must:
/// 1. Access the same memref with the same indices and vector type as the
/// transfer_read.
/// 2. Dominate the transfer_read operation.
/// If several candidates are available, one must be dominated by all the others
/// since they are all dominating the same transfer_read. We only consider the
/// transfer_write dominated by all the other candidates as this will be the
/// last transfer_write executed before the transfer_read.
/// If we found such a candidate we can do the forwarding if all the other
/// potentially aliasing ops that may reach the transfer_read are post-dominated
/// by the transfer_write.
void TransferOptimization::storeToLoadForwarding(vector::TransferReadOp read) {
  if (read.hasOutOfBoundsDim())
    return;
  LLVM_DEBUG(DBGS() << "Candidate for Forwarding: " << *read.getOperation()
                    << "\n");
  SmallVector<Operation *, 8> blockingWrites;
  vector::TransferWriteOp lastwrite = nullptr;
  Value source = memref::skipViewLikeOps(cast<MemrefValue>(read.getBase()));
  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    // If the user has already been processed skip.
    if (!processed.insert(user).second)
      continue;
    if (isa<ViewLikeOpInterface>(user)) {
      users.append(user->getUsers().begin(), user->getUsers().end());
      continue;
    }
    if (isMemoryEffectFree(user) || isa<vector::TransferReadOp>(user))
      continue;
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      // If there is a write, but we can prove that it is disjoint we can ignore
      // the write.
      if (vector::isDisjointTransferSet(
              cast<VectorTransferOpInterface>(write.getOperation()),
              cast<VectorTransferOpInterface>(read.getOperation()),
              /*testDynamicValueUsingBounds=*/true))
        continue;
      if (memref::isSameViewOrTrivialAlias(
              cast<MemrefValue>(read.getBase()),
              cast<MemrefValue>(write.getBase())) &&
          dominators.dominates(write, read) && checkSameValueRAW(write, read)) {
        if (lastwrite == nullptr || dominators.dominates(lastwrite, write))
          lastwrite = write;
        else
          assert(dominators.dominates(write, lastwrite));
        continue;
      }
    }
    blockingWrites.push_back(user);
  }

  if (lastwrite == nullptr)
    return;

  Region *topRegion = lastwrite->getParentRegion();
  Operation *readAncestor = findAncestorOpInRegion(topRegion, read);
  assert(readAncestor &&
         "read op should be recursively part of the top region");

  for (Operation *write : blockingWrites) {
    Operation *writeAncestor = findAncestorOpInRegion(topRegion, write);
    // TODO: if the store and read have the same ancestor we could recurse in
    // the region to know if the read is reachable with more precision.
    if (writeAncestor == nullptr || !isReachable(writeAncestor, readAncestor))
      continue;
    if (!postDominators.postDominates(lastwrite, write)) {
      LLVM_DEBUG(DBGS() << "Fail to do write to read forwarding due to op: "
                        << *write << "\n");
      return;
    }
  }

  LLVM_DEBUG(DBGS() << "Forward value from " << *lastwrite.getOperation()
                    << " to: " << *read.getOperation() << "\n");
  read.replaceAllUsesWith(lastwrite.getVector());
  opToErase.push_back(read.getOperation());
}

/// Converts OpFoldResults to int64_t shape without unit dims.
static SmallVector<int64_t> getReducedShape(ArrayRef<OpFoldResult> mixedSizes) {
  SmallVector<int64_t> reducedShape;
  for (const auto size : mixedSizes) {
    if (llvm::dyn_cast_if_present<Value>(size)) {
      reducedShape.push_back(ShapedType::kDynamic);
      continue;
    }

    auto value = cast<IntegerAttr>(cast<Attribute>(size)).getValue();
    if (value == 1)
      continue;
    reducedShape.push_back(value.getSExtValue());
  }
  return reducedShape;
}

/// Drops unit dimensions from the input MemRefType.
static MemRefType dropUnitDims(MemRefType inputType,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes,
                               ArrayRef<OpFoldResult> strides) {
  auto targetShape = getReducedShape(sizes);
  MemRefType rankReducedType = memref::SubViewOp::inferRankReducedResultType(
      targetShape, inputType, offsets, sizes, strides);
  return rankReducedType.canonicalizeStridedLayout();
}

/// Creates a rank-reducing memref.subview op that drops unit dims from its
/// input. Or just returns the input if it was already without unit dims.
static Value rankReducingSubviewDroppingUnitDims(PatternRewriter &rewriter,
                                                 mlir::Location loc,
                                                 Value input) {
  MemRefType inputType = cast<MemRefType>(input.getType());
  SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = memref::getMixedSizes(rewriter, loc, input);
  SmallVector<OpFoldResult> strides(inputType.getRank(),
                                    rewriter.getIndexAttr(1));
  MemRefType resultType = dropUnitDims(inputType, offsets, sizes, strides);

  if (resultType.canonicalizeStridedLayout() ==
      inputType.canonicalizeStridedLayout())
    return input;
  return memref::SubViewOp::create(rewriter, loc, resultType, input, offsets,
                                   sizes, strides);
}

/// Returns the number of dims that aren't unit dims.
static int getReducedRank(ArrayRef<int64_t> shape) {
  return llvm::count_if(shape, [](int64_t dimSize) { return dimSize != 1; });
}

/// Trims non-scalable one dimensions from `oldType` and returns the result
/// type.
static VectorType trimNonScalableUnitDims(VectorType oldType) {
  SmallVector<int64_t> newShape;
  SmallVector<bool> newScalableDims;
  for (auto [dimIdx, dimSize] : llvm::enumerate(oldType.getShape())) {
    if (dimSize == 1 && !oldType.getScalableDims()[dimIdx])
      continue;
    newShape.push_back(dimSize);
    newScalableDims.push_back(oldType.getScalableDims()[dimIdx]);
  }
  return VectorType::get(newShape, oldType.getElementType(), newScalableDims);
}

// Rewrites vector.create_mask 'op' to drop non-scalable one dimensions.
static FailureOr<Value>
createMaskDropNonScalableUnitDims(PatternRewriter &rewriter, Location loc,
                                  vector::CreateMaskOp op) {
  auto type = op.getType();
  VectorType reducedType = trimNonScalableUnitDims(type);
  if (reducedType.getRank() == type.getRank())
    return failure();

  SmallVector<Value> reducedOperands;
  for (auto [dim, dimIsScalable, operand] : llvm::zip_equal(
           type.getShape(), type.getScalableDims(), op.getOperands())) {
    if (dim == 1 && !dimIsScalable) {
      // If the mask for the unit dim is not a constant of 1, do nothing.
      auto constant = operand.getDefiningOp<arith::ConstantIndexOp>();
      if (!constant || (constant.value() != 1))
        return failure();
      continue;
    }
    reducedOperands.push_back(operand);
  }
  return vector::CreateMaskOp::create(rewriter, loc, reducedType,
                                      reducedOperands)
      .getResult();
}

namespace {

/// Rewrites `vector.transfer_read` ops where the source has unit dims, by
/// inserting a memref.subview dropping those unit dims. The vector shapes are
/// also reduced accordingly.
class TransferReadDropUnitDimsPattern
    : public vector::MaskableOpRewritePattern<vector::TransferReadOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::TransferReadOp transferReadOp,
                            vector::MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferReadOp.getBase();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());
    // TODO: support tensor types.
    if (!sourceType)
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim())
      return failure();
    if (!transferReadOp.getPermutationMap().isMinorIdentity())
      return failure();
    // Check if the source shape can be further reduced.
    int reducedRank = getReducedRank(sourceType.getShape());
    if (reducedRank == sourceType.getRank())
      return failure();
    // TODO: Extend vector.mask to support 0-d vectors. In the meantime, bail
    // out.
    if (reducedRank == 0 && maskingOp)
      return failure();
    // Check if the reduced vector shape matches the reduced source shape.
    // Otherwise, this case is not supported yet.
    VectorType reducedVectorType = trimNonScalableUnitDims(vectorType);
    if (reducedRank != reducedVectorType.getRank())
      return failure();
    if (llvm::any_of(transferReadOp.getIndices(), [](Value v) {
          return getConstantIntValue(v) != static_cast<int64_t>(0);
        }))
      return failure();

    Value maskOp = transferReadOp.getMask();
    if (maskOp) {
      auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
      if (!createMaskOp)
        return rewriter.notifyMatchFailure(
            transferReadOp, "unsupported mask op, only 'vector.create_mask' is "
                            "currently supported");
      FailureOr<Value> rankReducedCreateMask =
          createMaskDropNonScalableUnitDims(rewriter, loc, createMaskOp);
      if (failed(rankReducedCreateMask))
        return failure();
      maskOp = *rankReducedCreateMask;
    }

    Value reducedShapeSource =
        rankReducingSubviewDroppingUnitDims(rewriter, loc, source);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeros(reducedRank, c0);
    auto identityMap = rewriter.getMultiDimIdentityMap(reducedRank);
    SmallVector<bool> inBounds(reducedVectorType.getRank(), true);
    Operation *newTransferReadOp = vector::TransferReadOp::create(
        rewriter, loc, reducedVectorType, reducedShapeSource, zeros,
        identityMap, transferReadOp.getPadding(), maskOp,
        rewriter.getBoolArrayAttr(inBounds));

    if (maskingOp) {
      auto shapeCastMask = rewriter.createOrFold<vector::ShapeCastOp>(
          loc, reducedVectorType.cloneWith(std::nullopt, rewriter.getI1Type()),
          maskingOp.getMask());
      newTransferReadOp = mlir::vector::maskOperation(
          rewriter, newTransferReadOp, shapeCastMask);
    }

    auto shapeCast = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, vectorType, newTransferReadOp->getResults()[0]);

    return shapeCast;
  }
};

/// Rewrites `vector.transfer_write` ops where the "source" (i.e. destination)
/// has unit dims, by inserting a `memref.subview` dropping those unit dims. The
/// vector shapes are also reduced accordingly.
class TransferWriteDropUnitDimsPattern
    : public vector::MaskableOpRewritePattern<vector::TransferWriteOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::TransferWriteOp transferWriteOp,
                            vector::MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferWriteOp.getBase();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());
    // TODO: support tensor type.
    if (!sourceType)
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim())
      return failure();
    if (!transferWriteOp.getPermutationMap().isMinorIdentity())
      return failure();
    // Check if the destination shape can be further reduced.
    int reducedRank = getReducedRank(sourceType.getShape());
    if (reducedRank == sourceType.getRank())
      return failure();
    // TODO: Extend vector.mask to support 0-d vectors. In the meantime, bail
    // out.
    if (reducedRank == 0 && maskingOp)
      return failure();
    // Check if the reduced vector shape matches the reduced destination shape.
    // Otherwise, this case is not supported yet.
    VectorType reducedVectorType = trimNonScalableUnitDims(vectorType);
    if (reducedRank != reducedVectorType.getRank())
      return failure();
    if (llvm::any_of(transferWriteOp.getIndices(), [](Value v) {
          return getConstantIntValue(v) != static_cast<int64_t>(0);
        }))
      return failure();

    Value maskOp = transferWriteOp.getMask();
    if (maskOp) {
      auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
      if (!createMaskOp)
        return rewriter.notifyMatchFailure(
            transferWriteOp,
            "unsupported mask op, only 'vector.create_mask' is "
            "currently supported");
      FailureOr<Value> rankReducedCreateMask =
          createMaskDropNonScalableUnitDims(rewriter, loc, createMaskOp);
      if (failed(rankReducedCreateMask))
        return failure();
      maskOp = *rankReducedCreateMask;
    }
    Value reducedShapeSource =
        rankReducingSubviewDroppingUnitDims(rewriter, loc, source);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeros(reducedRank, c0);
    auto identityMap = rewriter.getMultiDimIdentityMap(reducedRank);
    SmallVector<bool> inBounds(reducedVectorType.getRank(), true);
    auto shapeCastSrc = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, reducedVectorType, vector);
    Operation *newXferWrite = vector::TransferWriteOp::create(
        rewriter, loc, Type(), shapeCastSrc, reducedShapeSource, zeros,
        identityMap, maskOp, rewriter.getBoolArrayAttr(inBounds));

    if (maskingOp) {
      auto shapeCastMask = rewriter.createOrFold<vector::ShapeCastOp>(
          loc, reducedVectorType.cloneWith(std::nullopt, rewriter.getI1Type()),
          maskingOp.getMask());
      newXferWrite =
          mlir::vector::maskOperation(rewriter, newXferWrite, shapeCastMask);
    }

    if (transferWriteOp.hasPureTensorSemantics())
      return newXferWrite->getResults()[0];

    // With Memref semantics, there's no return value. Use empty value to signal
    // success.
    return Value();
  }
};

} // namespace

/// Creates a memref.collapse_shape collapsing all inner dimensions of the
/// input starting at `firstDimToCollapse`.
static Value collapseInnerDims(PatternRewriter &rewriter, mlir::Location loc,
                               Value input, int64_t firstDimToCollapse) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  if (inputType.getRank() == 1)
    return input;
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < firstDimToCollapse; ++i)
    reassociation.push_back(ReassociationIndices{i});
  ReassociationIndices collapsedIndices;
  for (int64_t i = firstDimToCollapse; i < inputType.getRank(); ++i)
    collapsedIndices.push_back(i);
  reassociation.push_back(collapsedIndices);
  return memref::CollapseShapeOp::create(rewriter, loc, input, reassociation);
}

/// Returns the new indices that collapses the inner dimensions starting from
/// the `firstDimToCollapse` dimension.
static SmallVector<Value> getCollapsedIndices(RewriterBase &rewriter,
                                              Location loc,
                                              ArrayRef<int64_t> shape,
                                              ValueRange indices,
                                              int64_t firstDimToCollapse) {
  assert(firstDimToCollapse < static_cast<int64_t>(indices.size()));

  // If all the collapsed indices are zero then no extra logic is needed.
  // Otherwise, a new offset/index has to be computed.
  SmallVector<Value> indicesAfterCollapsing(
      indices.begin(), indices.begin() + firstDimToCollapse);
  SmallVector<Value> indicesToCollapse(indices.begin() + firstDimToCollapse,
                                       indices.end());
  if (llvm::all_of(indicesToCollapse, isZeroInteger)) {
    indicesAfterCollapsing.push_back(indicesToCollapse[0]);
    return indicesAfterCollapsing;
  }

  // Compute the remaining trailing index/offset required for reading from
  // the collapsed memref:
  //
  //    offset = 0
  //    for (i = firstDimToCollapse; i < outputRank; ++i)
  //      offset += sourceType.getDimSize(i) * transferReadOp.indices[i]
  //
  // For this example:
  //   %2 = vector.transfer_read/write %arg4[%c0, %arg0, %c0] (...) :
  //      memref<1x43x2xi32>, vector<1x2xi32>
  // which would be collapsed to:
  //   %1 = vector.transfer_read/write %collapse_shape[%c0, %offset] (...) :
  //      memref<1x86xi32>, vector<2xi32>
  // one would get the following offset:
  //    %offset = %arg0 * 43
  OpFoldResult collapsedOffset =
      arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();

  auto collapsedStrides = computeSuffixProduct(
      ArrayRef<int64_t>(shape.begin() + firstDimToCollapse, shape.end()));

  // Compute the collapsed offset.
  auto &&[collapsedExpr, collapsedVals] =
      computeLinearIndex(collapsedOffset, collapsedStrides, indicesToCollapse);
  collapsedOffset = affine::makeComposedFoldedAffineApply(
      rewriter, loc, collapsedExpr, collapsedVals);

  if (auto value = dyn_cast<Value>(collapsedOffset)) {
    indicesAfterCollapsing.push_back(value);
  } else {
    indicesAfterCollapsing.push_back(arith::ConstantIndexOp::create(
        rewriter, loc, *getConstantIntValue(collapsedOffset)));
  }

  return indicesAfterCollapsing;
}

namespace {
/// Rewrites contiguous row-major vector.transfer_read ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_read has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
///
/// If `targetVectorBitwidth` is provided, the flattening will only happen if
/// the trailing dimension of the vector read is smaller than the provided
/// bitwidth.
class FlattenContiguousRowMajorTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
public:
  FlattenContiguousRowMajorTransferReadPattern(MLIRContext *context,
                                               unsigned vectorBitwidth,
                                               PatternBenefit benefit)
      : OpRewritePattern<vector::TransferReadOp>(context, benefit),
        targetVectorBitwidth(vectorBitwidth) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    auto source = transferReadOp.getBase();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());

    // 0. Check pre-conditions
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    // If this is already 0D/1D, there's nothing to do.
    if (vectorType.getRank() <= 1)
      return failure();
    if (!vectorType.getElementType().isSignlessIntOrFloat())
      return failure();
    unsigned trailingVectorDimBitwidth =
        vectorType.getShape().back() * vectorType.getElementTypeBitWidth();
    if (trailingVectorDimBitwidth >= targetVectorBitwidth)
      return failure();
    if (!vector::isContiguousSlice(sourceType, vectorType))
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim())
      return failure();
    if (!transferReadOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (transferReadOp.getMask())
      return failure();

    // Determine the first memref dimension to collapse - just enough so we can
    // read a flattened vector.
    int64_t firstDimToCollapse =
        sourceType.getRank() -
        vectorType.getShape().drop_while([](auto v) { return v == 1; }).size();

    // 1. Collapse the source memref
    Value collapsedSource =
        collapseInnerDims(rewriter, loc, source, firstDimToCollapse);
    MemRefType collapsedSourceType =
        cast<MemRefType>(collapsedSource.getType());
    int64_t collapsedRank = collapsedSourceType.getRank();
    assert(collapsedRank == firstDimToCollapse + 1);

    // 2. Generate input args for a new vector.transfer_read that will read
    // from the collapsed memref.
    // 2.1. New dim exprs + affine map
    SmallVector<AffineExpr, 1> dimExprs{
        getAffineDimExpr(firstDimToCollapse, rewriter.getContext())};
    auto collapsedMap =
        AffineMap::get(collapsedRank, 0, dimExprs, rewriter.getContext());

    // 2.2 New indices
    SmallVector<Value> collapsedIndices =
        getCollapsedIndices(rewriter, loc, sourceType.getShape(),
                            transferReadOp.getIndices(), firstDimToCollapse);

    // 3. Create new vector.transfer_read that reads from the collapsed memref
    VectorType flatVectorType = VectorType::get({vectorType.getNumElements()},
                                                vectorType.getElementType());
    vector::TransferReadOp flatRead = vector::TransferReadOp::create(
        rewriter, loc, flatVectorType, collapsedSource, collapsedIndices,
        transferReadOp.getPadding(), collapsedMap);
    flatRead.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));

    // 4. Replace the old transfer_read with the new one reading from the
    // collapsed shape
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        transferReadOp, cast<VectorType>(vector.getType()), flatRead);
    return success();
  }

private:
  // Minimum bitwidth that the trailing vector dimension should have after
  // flattening.
  unsigned targetVectorBitwidth;
};

/// Rewrites contiguous row-major vector.transfer_write ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_write has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
///
/// If `targetVectorBitwidth` is provided, the flattening will only happen if
/// the trailing dimension of the vector read is smaller than the provided
/// bitwidth.
class FlattenContiguousRowMajorTransferWritePattern
    : public OpRewritePattern<vector::TransferWriteOp> {
public:
  FlattenContiguousRowMajorTransferWritePattern(MLIRContext *context,
                                                unsigned vectorBitwidth,
                                                PatternBenefit benefit)
      : OpRewritePattern<vector::TransferWriteOp>(context, benefit),
        targetVectorBitwidth(vectorBitwidth) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferWriteOp.getBase();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());

    // 0. Check pre-conditions
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    // If this is already 0D/1D, there's nothing to do.
    if (vectorType.getRank() <= 1)
      // Already 0D/1D, nothing to do.
      return failure();
    if (!vectorType.getElementType().isSignlessIntOrFloat())
      return failure();
    unsigned trailingVectorDimBitwidth =
        vectorType.getShape().back() * vectorType.getElementTypeBitWidth();
    if (trailingVectorDimBitwidth >= targetVectorBitwidth)
      return failure();
    if (!vector::isContiguousSlice(sourceType, vectorType))
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim())
      return failure();
    if (!transferWriteOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (transferWriteOp.getMask())
      return failure();

    // Determine the first memref dimension to collapse - just enough so we can
    // read a flattened vector.
    int64_t firstDimToCollapse =
        sourceType.getRank() -
        vectorType.getShape().drop_while([](auto v) { return v == 1; }).size();

    // 1. Collapse the source memref
    Value collapsedSource =
        collapseInnerDims(rewriter, loc, source, firstDimToCollapse);
    MemRefType collapsedSourceType =
        cast<MemRefType>(collapsedSource.getType());
    int64_t collapsedRank = collapsedSourceType.getRank();
    assert(collapsedRank == firstDimToCollapse + 1);

    // 2. Generate input args for a new vector.transfer_read that will read
    // from the collapsed memref.
    // 2.1. New dim exprs + affine map
    SmallVector<AffineExpr, 1> dimExprs{
        getAffineDimExpr(firstDimToCollapse, rewriter.getContext())};
    auto collapsedMap =
        AffineMap::get(collapsedRank, 0, dimExprs, rewriter.getContext());

    // 2.2 New indices
    SmallVector<Value> collapsedIndices =
        getCollapsedIndices(rewriter, loc, sourceType.getShape(),
                            transferWriteOp.getIndices(), firstDimToCollapse);

    // 3. Create new vector.transfer_write that writes to the collapsed memref
    VectorType flatVectorType = VectorType::get({vectorType.getNumElements()},
                                                vectorType.getElementType());
    Value flatVector =
        vector::ShapeCastOp::create(rewriter, loc, flatVectorType, vector);
    vector::TransferWriteOp flatWrite = vector::TransferWriteOp::create(
        rewriter, loc, flatVector, collapsedSource, collapsedIndices,
        collapsedMap);
    flatWrite.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));

    // 4. Replace the old transfer_write with the new one writing the
    // collapsed shape
    rewriter.eraseOp(transferWriteOp);
    return success();
  }

private:
  // Minimum bitwidth that the trailing vector dimension should have after
  // flattening.
  unsigned targetVectorBitwidth;
};

/// Rewrite `vector.extract(vector.transfer_read)` to `memref.load`.
///
/// All the users of the transfer op must be `vector.extract` ops. If
/// `allowMultipleUses` is set to true, rewrite transfer ops with any number of
/// users. Otherwise, rewrite only if the extract op is the single user of the
/// transfer op. Rewriting a single vector load with multiple scalar loads may
/// negatively affect performance.
class RewriteScalarExtractOfTransferRead
    : public OpRewritePattern<vector::ExtractOp> {
public:
  RewriteScalarExtractOfTransferRead(MLIRContext *context,
                                     PatternBenefit benefit,
                                     bool allowMultipleUses)
      : OpRewritePattern(context, benefit),
        allowMultipleUses(allowMultipleUses) {}

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Match phase.
    auto xferOp = extractOp.getVector().getDefiningOp<vector::TransferReadOp>();
    if (!xferOp)
      return failure();
    // Check that we are extracting a scalar and not a sub-vector.
    if (isa<VectorType>(extractOp.getResult().getType()))
      return failure();
    // If multiple uses are not allowed, check if xfer has a single use.
    if (!allowMultipleUses && !xferOp.getResult().hasOneUse())
      return failure();
    // If multiple uses are allowed, check if all the xfer uses are extract ops.
    if (allowMultipleUses &&
        !llvm::all_of(xferOp->getUses(), [](OpOperand &use) {
          return isa<vector::ExtractOp>(use.getOwner());
        }))
      return failure();
    // Mask not supported.
    if (xferOp.getMask())
      return failure();
    // Map not supported.
    if (!xferOp.getPermutationMap().isMinorIdentity())
      return failure();
    // Cannot rewrite if the indices may be out of bounds.
    if (xferOp.hasOutOfBoundsDim())
      return failure();

    // Rewrite phase: construct scalar load.
    SmallVector<Value> newIndices(xferOp.getIndices().begin(),
                                  xferOp.getIndices().end());
    for (auto [i, pos] : llvm::enumerate(extractOp.getMixedPosition())) {
      int64_t idx = newIndices.size() - extractOp.getNumIndices() + i;

      // Compute affine expression `newIndices[idx] + pos` where `pos` can be
      // either a constant or a value.
      OpFoldResult composedIdx;
      if (auto attr = dyn_cast<Attribute>(pos)) {
        int64_t offset = cast<IntegerAttr>(attr).getInt();
        composedIdx = affine::makeComposedFoldedAffineApply(
            rewriter, extractOp.getLoc(),
            rewriter.getAffineSymbolExpr(0) + offset, {newIndices[idx]});
      } else {
        Value dynamicOffset = cast<Value>(pos);
        AffineExpr sym0, sym1;
        bindSymbols(rewriter.getContext(), sym0, sym1);
        composedIdx = affine::makeComposedFoldedAffineApply(
            rewriter, extractOp.getLoc(), sym0 + sym1,
            {newIndices[idx], dynamicOffset});
      }

      // Update the corresponding index with the folded result.
      if (auto value = dyn_cast<Value>(composedIdx)) {
        newIndices[idx] = value;
      } else {
        newIndices[idx] = arith::ConstantIndexOp::create(
            rewriter, extractOp.getLoc(), *getConstantIntValue(composedIdx));
      }
    }
    if (isa<MemRefType>(xferOp.getBase().getType())) {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(extractOp, xferOp.getBase(),
                                                  newIndices);
    } else {
      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
          extractOp, xferOp.getBase(), newIndices);
    }

    return success();
  }

private:
  bool allowMultipleUses;
};

/// Rewrite transfer_writes of vectors of size 1 (e.g., vector<1x1xf32>)
/// to memref.store.
class RewriteScalarWrite : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override {
    // Must be a scalar write.
    auto vecType = xferOp.getVectorType();
    if (!llvm::all_of(vecType.getShape(), [](int64_t sz) { return sz == 1; }))
      return failure();
    // Mask not supported.
    if (xferOp.getMask())
      return failure();
    // Map not supported.
    if (!xferOp.getPermutationMap().isMinorIdentity())
      return failure();
    // Only float and integer element types are supported.
    Value scalar = vector::ExtractOp::create(rewriter, xferOp.getLoc(),
                                             xferOp.getVector());
    // Construct a scalar store.
    if (isa<MemRefType>(xferOp.getBase().getType())) {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(
          xferOp, scalar, xferOp.getBase(), xferOp.getIndices());
    } else {
      rewriter.replaceOpWithNewOp<tensor::InsertOp>(
          xferOp, scalar, xferOp.getBase(), xferOp.getIndices());
    }
    return success();
  }
};

} // namespace

void mlir::vector::transferOpflowOpt(RewriterBase &rewriter,
                                     Operation *rootOp) {
  TransferOptimization opt(rewriter, rootOp);
  // Run store to load forwarding first since it can expose more dead store
  // opportunity.
  rootOp->walk([&](vector::TransferReadOp read) {
    if (isa<MemRefType>(read.getShapedType()))
      opt.storeToLoadForwarding(read);
  });
  opt.removeDeadOp();
  rootOp->walk([&](vector::TransferWriteOp write) {
    if (isa<MemRefType>(write.getShapedType()))
      opt.deadStoreOp(write);
  });
  opt.removeDeadOp();
}

void mlir::vector::populateScalarVectorTransferLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit,
    bool allowMultipleUses) {
  patterns.add<RewriteScalarExtractOfTransferRead>(patterns.getContext(),
                                                   benefit, allowMultipleUses);
  patterns.add<RewriteScalarWrite>(patterns.getContext(), benefit);
}

void mlir::vector::populateVectorTransferDropUnitDimsPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<TransferReadDropUnitDimsPattern, TransferWriteDropUnitDimsPattern>(
          patterns.getContext(), benefit);
}

void mlir::vector::populateFlattenVectorTransferPatterns(
    RewritePatternSet &patterns, unsigned targetVectorBitwidth,
    PatternBenefit benefit) {
  patterns.add<FlattenContiguousRowMajorTransferReadPattern,
               FlattenContiguousRowMajorTransferWritePattern>(
      patterns.getContext(), targetVectorBitwidth, benefit);
  populateDropUnitDimWithShapeCastPatterns(patterns, benefit);
}
