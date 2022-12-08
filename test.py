class MinkowskiConvolution(MinkowskiConvolutionBase):
    r"""Convolution layer for a sparse tensor.


    .. math::

        \mathbf{x}_\mathbf{u} = \sum_{\mathbf{i} \in \mathcal{N}^D(\mathbf{u}, K,
        \mathcal{C}^\text{in})} W_\mathbf{i} \mathbf{x}_{\mathbf{i} +
        \mathbf{u}} \;\text{for} \; \mathbf{u} \in \mathcal{C}^\text{out}

    where :math:`K` is the kernel size and :math:`\mathcal{N}^D(\mathbf{u}, K,
    \mathcal{C}^\text{in})` is the set of offsets that are at most :math:`\left
    \lceil{\frac{1}{2}(K - 1)} \right \rceil` away from :math:`\mathbf{u}`
    definied in :math:`\mathcal{S}^\text{in}`.

    .. note::
        For even :math:`K`, the kernel offset :math:`\mathcal{N}^D`
        implementation is different from the above definition. The offsets
        range from :math:`\mathbf{i} \in [0, K)^D, \; \mathbf{i} \in
        \mathbb{Z}_+^D`.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        #卷积核形状
        expand_coordinates=False,
        #强制生成新坐标。当为True时，输出坐标将是内核形状和输入坐标的外积。
        convolution_mode=ConvolutionMode.DEFAULT,
        dimension=None,
    ):
        r"""convolution on a sparse tensor

        Args:
            :attr:`in_channels` (int): the number of input channels in the
            input tensor.

            :attr:`out_channels` (int): the number of output channels in the
            output tensor.

            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size of the
            convolution layer. If non-identity is used, the output coordinates
            will be at least :attr:`stride` :math:`\times` :attr:`tensor_stride`
            away. When a list is given, the length must be D; each element will
            be used for stride size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines custom kernel shape.

            :attr:`expand_coordinates` (bool, optional): Force generation of
            new coordinates. When True, the output coordinates will be the
            outer product of the kernel shape and the input coordinates.
            `False` by default.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        """
        MinkowskiConvolutionBase.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            kernel_generator,
            is_transpose=False,
            expand_coordinates=expand_coordinates,
            convolution_mode=convolution_mode,
            dimension=dimension,
        )
        self.reset_parameters()

        fw_fn = get_minkowski_function("ConvolutionForward", input_features)
        return fw_fn(
            ctx.input_features,
            kernel_weights,
            kernel_generator.kernel_size,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_dilation,
            kernel_generator.region_type,
            kernel_generator.region_offsets,
            kernel_generator.expand_coordinates,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager._manager,
        )



#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor ConvolutionForwardGPU(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    bool const expand_coordinates,                     //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);