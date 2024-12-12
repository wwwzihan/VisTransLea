"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial # 创建一个新的可调用的函数，简化函数调用
from collections import OrderedDict # 导入有序字典数据结构类型

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    """ 参数：
    x:输入的张量, 代表神经网络某一层的输入数据.
    drop_prob: 丢弃路径的概率, 默认值为0，即不进行丢弃操作。
    training: 用于指示当前是否处于训练模式，只有在训练模式且drop_prob不为0时才执行路径丢弃操作。
    """
    # 首先进行条件判断，如果丢弃率等于0或者当前不是处于训练模式，那么直接返回输入的张量x，不进行任何操作
    if drop_prob == 0. or not training:
        return x
    # 当满足丢弃条件时，计算保留概率
    keep_prob = 1 - drop_prob
    # 根据输入张量x的维度信息来构造一个新的元组shape，这个新的元组将用于后续生成与输入张量x在维度上相匹配的随机张量
    # x.shape[0]表示张量x的第一个特征维度的大小，这个特征通常表示样本数量
    # x.ndim获取张量x的维度，减去第一个被提取出来的维度，即重复1(x.ndim-1)次，然后将元祖进行拼接。
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 生成一个与shape特征维度相同的随机张量，与keep_prob相加，仅对特征值进行改变，维度不变。与keep_prob相加后范围变为(keep_prob, 1+keep_prob)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # floor_()方法用于将张量中的每个元素向下取整到最接近的整数。这里的原地（通过末尾的下划线_表示）意味着该操作会直接修改random_tensor本身，而不是返回一个新的取整后的张量副本。
    random_tensor.floor_()  # binarize
    # 根据之前生成并经过处理的随机张量来对输入张量x进行相应的调整，最终得到处理后的输出张量output
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    # drop_prob是一个可选的参数，如果在创建DropPath类的实例时没有传入这个参数，那么其将被初始化为None
    def __init__(self, drop_prob=None):
        # 初始化操作，调用父类的构造函数，确保DropPath类继承了nn.Module的所有属性和方法，基操
        super(DropPath, self).__init__()
        # 将传入的drop_prob参数赋值给类的实例属性，便于在后续的前向传播过程中能够使用这个指定的丢弃路径概率
        self.drop_prob = drop_prob

    # 类的前向传播函数，也称为类的方法，通过操作实例属性来实现特定的功能
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    将二维图像转换为图像块(patch)的嵌入表示，有助于将图像数据处理成适合Transformer处理的格式
    """
    # __init__是类的构造函数
    """ 参数说明
    img_size：输入图像的尺寸，这里将其转换为一个二元组(img_size,img_size)的形式存储在类的实例属性中
    patch_size：划分图像后得到的每个图像块的尺寸
    in_c：输入图像的通道数
    embed_dim：图像块映射到特征空间的维度
    norm_layer：可选的归一化层类型
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        # 初始化，调用父类的构造函数
        super().__init__()
        # 将img_size和patch_size转换为二元组形式存储
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        # 将输入图像尺寸和图像块尺寸的二元组分别赋值给类的实例属性
        self.img_size = img_size
        self.patch_size = patch_size
        # grid:网格，计算在水平和竖直方向上图像可以划分成的图像块数量，得到一个表示图像块网格尺寸的二元组，并赋值给类的实例属性
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 得到总的图像块的数量
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 创建一个二维卷积层，参数：输入通道、输出通道、卷积核尺寸、步长
        # 该卷积操作的作用：将输入图像按照图像块的尺寸进行卷积操作，从而将每个图像块映射到embed_dim维度的特征空间，实现图像到图像块嵌入表示的初步转换
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 如果norm_layer存在，则创建相应的归一化层并传入embed_dim作为参数，对经过卷积操作后的图像块嵌入表示进行归一化；如果不存在，则保持数据原样通过
        """ 归一化操作的意义：
        1 归一化将数据映射到一个相对较小切固定的范围，使得梯度下降算法能够更稳定、更快的收敛。
        2 归一化有助于减少数据中存在的内部协变量偏移，使每一层的输入数据分布保持相对稳定。
        """
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    # 类的前向传播函数
    def forward(self, x):
        # 首先获取输入张量的形状信息，批次大小B、通道数C、图像高度H、图像宽度W
        B, C, H, W = x.shape
        # 进行输入验证，确保输入图像的高度和宽度与类实例中存储的预期图像尺寸相匹配，否则给出错误提示
        # '\'用于代码的换行延续
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]  将图像块的数量相关维度放在前面，嵌入维度放在后面
        # 首先利用构造函数创建的卷积层对输入图像进行卷积操作，然后将张量在第三个维度上进行扁平化操作，在然后将扁平化后的张量的第二个维度和第三个维度进行交换
        x = self.proj(x).flatten(2).transpose(1, 2)
        # 调用归一化类属性——执行归一化操作
        x = self.norm(x)
        return x

"""
1 transformer 中的 multi-head self-attention 模块
2 Attention类继承自nn.Module模块，是实现注意力机制的核心模块。它接受输入的特征表示，并通过一系列操作计算出经过注意力机制后的输出特征表示，
  用于捕获输入数据不同部分之间的相关性。
"""
class Attention(nn.Module):
    # 类的构造函数
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,             # 多头注意力机制中头的数量，多头注意力机制可以从不同的表示子空间中捕捉信息，增强模型的表达能力
                 qkv_bias=False,          # query查询、key键、value值 是否使用偏置
                 qk_scale=None,           # 可选项，用于指定q和k相乘时的缩放因子，此处未指定。
                 attn_drop_ratio=0.,      # 注意力权重丢弃率，用于防止过拟合
                 proj_drop_ratio=0.):     # 输出投影层的丢弃率
        super(Attention, self).__init__() # 调用Attention类的父类nn.Module的__init__方法
        self.num_heads = num_heads                          # 直接存储传入的头的数量
        head_dim = dim // num_heads                         # 通过将输入维度除以头的数量，来计算每个头所对应的维度。是后续qkv分别进行维度划分的基础
        self.scale = qk_scale or head_dim ** -0.5           # 确定查询和键相乘时的缩放因子，如果qk_scale有传入值则使用传入值，否则使用head_dim的负0.5次方作为缩放因子，\
                                                            # \这个缩放因子有助于调整注意力权重的计算，避免数值过大或过小。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # 创建一个线性层，输入维度为dim，输出维度为dim*3，这是因为在多头注意力机制中，需要同时生成\
                                                            # \查询、键、值，所以输出维度是输入维度的三倍
        self.attn_drop = nn.Dropout(attn_drop_ratio)        # 创建一个丢弃层，其丢弃率为attn_drop_ratio，用于在计算出注意力权重后进行随机丢弃操作
        self.proj = nn.Linear(dim, dim)                     # 这个线性层用于对经过注意力机制处理后的输出进行投影操作，将其映射回原始的输入维度，以便后续\
                                                            # \与其他模块进行连接等操作
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head] 维度重排
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        """ transpose转置操作的目的：
        对键k进行转置操作，在张量操作中，transpose用于交换指定的两个维度，@ 符号表示矩阵乘法，这里-2，-1表示倒数第二个和倒数第一个维度，这样做的目的是为了后续能\
        和查询q做矩阵乘法，q: [num_patches + 1, embed_dim_per_head] * k.t: [embed_dim_per_head, num_patches + 1]，= [num_patches + 1, num_patches + 1]，\
        这个结果可以理解为每个头(num_heads)下，每个 num_patches+1 个输入单元（如图像块），与其他num_patches+1个输入单元（图像块）之间的一种关联程度的矩阵表示。\
        在计算出查询与键的乘积后，乘以一个缩放因子，目的是调整计算出来的关联程度的值，避免在后续的计算中出现数值过大或过小的情况。
        """
        attn = (q @ k.transpose(-2, -1)) * self.scale
        """ softmax()操作的意义：
        对经过缩放后的结果的张量进行softmax操作，并且指定在最后一个维度上进行（实际是对每一行的元素进行softmax处理）。softmax操作会将张量中的每个元素转换为一个在0到1之间的概率值，并且使得在最后一个维度上\
        所有的元素的和为1，这样就将前面计算出的关联程度矩阵转换为了一个概率分布矩阵，每个元素表示对应位置的输入单元与其他输出单元之间的相对重要性（以概率的形式表示），\
        也就是得到了真正的注意力权重矩阵。
        """
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        """ reshape()方法
        reshape()方法实际上是对每个head进行concatenate拼接；
        transpose(1, 2)，这样做的目的是为了后续能方便地将其重塑为与输入张量 x 类似的维度结构，以便进行后续的处理和与其他模块的衔接；
        C 表示总的嵌入维度（total_embed_dim）
        """
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    """ 类的构造函数
    传入参数：输入特征，隐藏层（中间输出特征），输出特征，激活函数，丢弃率；
    实例属性：全连接属性，激活函数属性，丢弃层属性，初始化时直接构造，便于在forward计算中进行调用。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # 初始化，调用当前类的父类的__init__方法
        super().__init__()
        out_features = out_features or in_features  # 如果有传入out_features就等于传入的，如果没传入就等于in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层，输入节点个数，输出节点个数
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    # 类的构造函数（初始化方法）
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,          # 第一个全连接层节点个数是输入节点个数的四倍。
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,         # multi-head self-attention模块中最后全连接层后使用的drop_ratio
                 attn_drop_ratio=0.,    # softmax((q*k_transpose)/(d**-2))后的attn_drop_ratio
                 drop_path_ratio=0.,    # multi-head attention 和 MLP Block 后的 drop_path_ratio
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        # 初始化，调用当前类的父类的初始化方法
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio) # mlp_block中的hidden_dim是输入dim的4（mlp_ratio）倍
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) # 块1
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 块2
        return x

""" VisionTransformer类
该类实现ViT模型的架构，通过将图像分割成patches，并利用Transformer的自注意力机制来处理这些patches以进行特征提取和分类等操作
"""
class VisionTransformer(nn.Module):
    # 类的构造函数
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer，也就是Transformer Encoder当中重复堆叠Encoder Block的次数
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  #定义一个可学习的位置嵌入张量，用于为每个patch和token提供位置信息，初始化为全零张量
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
