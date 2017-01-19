LongArray
用int index获取long value的数组
这个类提供了一个长度可以达到int范围的数组,也就是说这个数组的下标有long,2^32这么大
然后,注意! 源码中说了,某些vm不允许开long这么大的数组,所以理论上是一个data[2^32]数组,实现的时候是一个data[2^2][2^30]数组.
这个从long -> int,int 的方法是 类中的 o 和 i 两个方法. 原理是分别取一个long类型数的前30位和后30位.
size:已经在数组set过val的最大的long index
ensureCapacity() 如果容量不够就扩容

CustomWidthArray

LOG2_BITS_PER_WORD = 6 2^6 = 64, 表示一个word由64bits表示,即一个long
numLongs() 右移6位就是除以64, 计算有多少个long
