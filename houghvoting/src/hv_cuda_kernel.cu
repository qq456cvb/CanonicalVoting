#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_math.h"
#include <thrust/device_vector.h>

#include <vector>
#include <iostream>


template <typename scalar_t>
__global__ void hv_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz_labels,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_labels,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> obj_labels,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grid_obj,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid_rot,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid_scale,
    float3 corner,
    const float* __restrict__ res,
    const int* __restrict__ num_rots)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < points.size(0)) {
        
        scalar_t objness = obj_labels[c];
        float3 corr = make_float3(
            xyz_labels[c][0] * scale_labels[c][0],
            xyz_labels[c][1] * scale_labels[c][1],
            xyz_labels[c][2] * scale_labels[c][2]
        );
        float3 point = make_float3(points[c][0], points[c][1], points[c][2]);
        const float rot_interval = 2 * 3.141592654f / (*num_rots);
        for (int i = 0; i < (*num_rots); i++) {
            float theta = i * rot_interval;
            float3 offset = make_float3(-cos(theta) * corr.x + sin(theta) * corr.z,
                -corr.y, -sin(theta) * corr.x - cos(theta) * corr.z);
            float3 center_grid = (point + offset - corner) / (*res);
            if (center_grid.x < 0 || center_grid.y < 0 || center_grid.z < 0 || 
                center_grid.x >= grid_obj.size(0) - 1 || center_grid.y >= grid_obj.size(1) - 1 || center_grid.z >= grid_obj.size(2) - 1) {
                continue;
            }
            int3 center_grid_floor = make_int3(center_grid);
            int3 center_grid_ceil = center_grid_floor + 1;
            float3 residual = fracf(center_grid);
            
            float3 w0 = 1.f - residual;
            float3 w1 = residual;
            
            float lll = w0.x * w0.y * w0.z * objness;
            float llh = w0.x * w0.y * w1.z * objness;
            float lhl = w0.x * w1.y * w0.z * objness;
            float lhh = w0.x * w1.y * w1.z * objness;
            float hll = w1.x * w0.y * w0.z * objness;
            float hlh = w1.x * w0.y * w1.z * objness;
            float hhl = w1.x * w1.y * w0.z * objness;
            float hhh = w1.x * w1.y * w1.z * objness;

            atomicAdd(&grid_obj[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z], lll);
            atomicAdd(&grid_obj[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z], llh);
            atomicAdd(&grid_obj[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z], lhl);
            atomicAdd(&grid_obj[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z], lhh);
            atomicAdd(&grid_obj[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z], hll);
            atomicAdd(&grid_obj[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z], hlh);
            atomicAdd(&grid_obj[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z], hhl);
            atomicAdd(&grid_obj[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z], hhh);

            float rot_vec[2] = {cos(theta), sin(theta)};
            for (int j = 0; j < 2; j++) {
                float rot = rot_vec[j];
                atomicAdd(&grid_rot[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z][j], lll * rot);
                atomicAdd(&grid_rot[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z][j], llh * rot);
                atomicAdd(&grid_rot[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z][j], lhl * rot);
                atomicAdd(&grid_rot[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z][j], lhh * rot);
                atomicAdd(&grid_rot[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z][j], hll * rot);
                atomicAdd(&grid_rot[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z][j], hlh * rot);
                atomicAdd(&grid_rot[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z][j], hhl * rot);
                atomicAdd(&grid_rot[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z][j], hhh * rot);
            }

            for (int j = 0; j < 3; j++) {
                float scale = scale_labels[c][j];
                atomicAdd(&grid_scale[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z][j], lll * scale);
                atomicAdd(&grid_scale[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z][j], llh * scale);
                atomicAdd(&grid_scale[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z][j], lhl * scale);
                atomicAdd(&grid_scale[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z][j], lhh * scale);
                atomicAdd(&grid_scale[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z][j], hll * scale);
                atomicAdd(&grid_scale[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z][j], hlh * scale);
                atomicAdd(&grid_scale[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z][j], hhl * scale);
                atomicAdd(&grid_scale[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z][j], hhh * scale);
            }
            
        }
    }
}


template <typename scalar_t>
__global__ void hv_cuda_average_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grid,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid_rot,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid_scale)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid.size(0) || y >= grid.size(1) || z >= grid.size(2)) return;

    float w = grid[x][y][z];
    for (int j = 0; j < 2; j++) {
        grid_rot[x][y][z][j] /= w + 1e-7;
    }
    for (int j = 0; j < 3; j++) {
        grid_scale[x][y][z][j] /= w + 1e-7;
    }
}

std::vector<torch::Tensor> hv_cuda_forward(
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots) 
{
    auto corners = torch::stack({std::get<0>(torch::min(points, 0)), std::get<0>(torch::max(points, 0))}, 0);  // 2 x 3
    auto corner = corners[0];  // 3
    auto diff = (corners[1] - corners[0]) / res;  // 3
    auto grid_obj = torch::zeros({diff[0].item().to<int>() + 1, diff[1].item().to<int>() + 1, diff[2].item().to<int>() + 1}, points.options());
    auto grid_rot = torch::zeros({diff[0].item().to<int>() + 1, diff[1].item().to<int>() + 1, diff[2].item().to<int>() + 1, 2}, points.options());
    auto grid_scale = torch::zeros({diff[0].item().to<int>() + 1, diff[1].item().to<int>() + 1, diff[2].item().to<int>() + 1, 3}, points.options());
    
    // std::cout << grid.size(0) << ", " << grid.size(1) << ", " << grid.size(2) << std::endl;
    // std::cout << corner << std::endl;
    
    const int threads = 1024;
    const dim3 blocks((points.size(0) + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(points.type(), "hv_forward_cuda", ([&] {
        hv_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            xyz_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            scale_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            obj_labels.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            grid_obj.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grid_rot.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grid_scale.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            make_float3(corner[0].item().to<float>(), corner[1].item().to<float>(), corner[2].item().to<float>()),
            res.data<float>(),
            num_rots.data<int>()
        );
      }));

    AT_DISPATCH_FLOATING_TYPES(points.type(), "hv_average_cuda", ([&] {
        hv_cuda_average_kernel<scalar_t><<<dim3((grid_obj.size(0) + 7) / 8, (grid_obj.size(1) + 7) / 8, (grid_obj.size(2) + 7) / 8), dim3(8, 8, 8)>>>(
            grid_obj.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grid_rot.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grid_scale.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));
    return {grid_obj, grid_rot, grid_scale};
}


template <typename scalar_t>
__global__ void hv_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_grid,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz_labels,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_labels,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> obj_labels,
    // torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_points,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_xyz_labels,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_scale_labels,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_obj_labels,
    float3 corner,
    const float* __restrict__ res,
    const int* __restrict__ num_rots)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < points.size(0)) {
        
        scalar_t objness = obj_labels[c];
        float3 corr = make_float3(
            xyz_labels[c][0] * scale_labels[c][0],
            xyz_labels[c][1] * scale_labels[c][1],
            xyz_labels[c][2] * scale_labels[c][2]
        );
        float3 point = make_float3(points[c][0], points[c][1], points[c][2]);
        float rot_interval = 2 * 3.141592654f / (*num_rots);
        for (int i = 0; i < (*num_rots); i++) {
            float theta = i * rot_interval;
            float3 offset = make_float3(-cos(theta) * corr.x + sin(theta) * corr.z,
                -corr.y, -sin(theta) * corr.x - cos(theta) * corr.z);
            float3 center_grid = (point + offset - corner) / (*res);
            if (center_grid.x < 0 || center_grid.y < 0 || center_grid.z < 0 || 
                center_grid.x >= grad_grid.size(0) - 1 || center_grid.y >= grad_grid.size(1) - 1 || center_grid.z >= grad_grid.size(2) - 1) {
                continue;
            }
            int3 center_grid_floor = make_int3(center_grid);
            int3 center_grid_ceil = center_grid_floor + 1;
            float3 residual = fracf(center_grid);
            
            float3 w0 = 1.f - residual;
            float3 w1 = residual;
            
            d_obj_labels[c] += grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z] * w0.x * w0.y * w0.z;
            d_obj_labels[c] += grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z] * w0.x * w0.y * w1.z;
            d_obj_labels[c] += grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z] * w0.x * w1.y * w0.z;
            d_obj_labels[c] += grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z] * w0.x * w1.y * w1.z;
            d_obj_labels[c] += grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z] * w1.x * w0.y * w0.z;
            d_obj_labels[c] += grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z] * w1.x * w0.y * w1.z;
            d_obj_labels[c] += grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z] * w1.x * w1.y * w0.z;
            d_obj_labels[c] += grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z] * w1.x * w1.y * w1.z;

            float3 dgrid_dcenter = make_float3(
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z] * w0.y * w0.z
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z] * w0.y * w1.z
                - grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z] * w1.y * w0.z
                - grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z] * w1.y * w1.z
                + grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z] * w0.y * w0.z
                + grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z] * w0.y * w1.z
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z] * w1.y * w0.z
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z] * w1.y * w1.z,
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z] * w0.x * w0.z
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z] * w0.x * w1.z
                + grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z] * w0.x * w0.z
                + grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z] * w0.x * w1.z
                - grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z] * w1.x * w0.z
                - grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z] * w1.x * w1.z
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z] * w1.x * w0.z
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z] * w1.x * w1.z,
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z] * w0.x * w0.y
                + grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z] * w0.x * w0.y
                - grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z] * w0.x * w1.y
                + grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z] * w0.x * w1.y
                - grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z] * w1.x * w0.y
                + grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z] * w1.x * w0.y
                - grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z] * w1.x * w1.y
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z] * w1.x * w1.y) * objness;
            
            // d_points[c][0] += dgrid_dcenter.x;
            // d_points[c][1] += dgrid_dcenter.y;
            // d_points[c][2] += dgrid_dcenter.z;

            float3 d_corr = make_float3(- cos(theta) * dgrid_dcenter.x - sin(theta) * dgrid_dcenter.z,
                -dgrid_dcenter.y, sin(theta) * dgrid_dcenter.x - cos(theta) * dgrid_dcenter.z);

            d_xyz_labels[c][0] += d_corr.x * scale_labels[c][0];
            d_xyz_labels[c][1] += d_corr.y * scale_labels[c][1];
            d_xyz_labels[c][2] += d_corr.z * scale_labels[c][2];

            d_scale_labels[c][0] += d_corr.x * xyz_labels[c][0];
            d_scale_labels[c][1] += d_corr.y * xyz_labels[c][1];
            d_scale_labels[c][2] += d_corr.z * xyz_labels[c][2];
        }
    }
}



std::vector<torch::Tensor> hv_cuda_backward(
    torch::Tensor grad_grid,
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots) 
{
    auto corners = torch::stack({std::get<0>(torch::min(points, 0)), std::get<0>(torch::max(points, 0))}, 0);  // 2 x 3
    auto corner = corners[0];  // 3
    auto diff = (corners[1] - corners[0]) / res;  // 3
    // auto d_points = torch::zeros_like(points);
    auto d_xyz_labels = torch::zeros_like(xyz_labels);
    auto d_scale_labels = torch::zeros_like(scale_labels);
    auto d_obj_labels = torch::zeros_like(obj_labels);
    
    const int threads = 512;
    const dim3 blocks((points.size(0) + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(points.type(), "hv_backward_cuda", ([&] {
        hv_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_grid.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            xyz_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            scale_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            obj_labels.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            // d_points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_xyz_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_scale_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_obj_labels.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            make_float3(corner[0].item().to<float>(), corner[1].item().to<float>(), corner[2].item().to<float>()),
            res.data<float>(),
            num_rots.data<int>()
        );
      }));
    return {d_xyz_labels, d_scale_labels, d_obj_labels};
}
