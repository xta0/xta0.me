---
layout: post
title: 分析TCP的三步握手
tag: TCP/IP
categories: 随笔

---

<em>所有文章均为作者原创，转载请注明出处</em>

突然想看一下TCP connection的具体过程，简单写了一个TCP server（python的twist简单而快速），和iOS的client（socket+NSStream）。配合wire shark来做一些分析：

<h2>建立三步握手</h2>

- SYN

C(client)向S（server）发送请求，SYN标志位为1，产生一个随机的32bit Seq Num:0x0b97d19c，TCP header如下：

```
Source port: 54133 (54133)
Destination port: mit-dov (91)
[Stream index: 0]
Sequence number: 0x0b97d19c
Header length: 44 bytes
Flags: 0x002 (SYN)
Window size value: 65535
[Calculated window size: 65535]
Checksum: 0xfc9c [validation disabled]
```



- ACK+SYN

S收到后，将SYN和ACK标志位至为1，同时产生一个新的Seq Num：0x0f8baa6e和一个Ack Num：0x0b97d19d（C的Seq Num+1）

```
Source port: mit-dov (91)
Destination port: 54133 (54133)
[Stream index: 0]
Sequence number: 0x0f8baa6e    
Acknowledgment number: 0x0b97d19d
Header length: 44 bytes
Flags: 0x012 (SYN, ACK)
Window size value: 65535
[Calculated window size: 65535]
Checksum: 0xb0cc [validation disabled]
```

- ACK

C收到后，先校验Seq Num对不对，如果正确，将ACK标志位至为1，同时产生新的Ack Num为0x0f8baa6f(S的Seq Num+1)。

```
Source port: 54133 (54133)
Destination port: mit-dov (91)
[Stream index: 0]
Sequence number: 0x0b97d19d
Acknowledgment: 0x0f8baa6f
Header length: 32 bytes
Flags: 0x010 (ACK)
Window size value: 8235
[Calculated window size: 131760]
[Window size scaling factor: 16]
Checksum: 0xcf36 [validation disabled]
```

至此，三步握手完成,链接建立，但是我们发现，在第三步的时候，C向S通知了自己window窗的大小：8235和 scaling factor：16。关于流控和滑动窗口不在这里讨论，但是当S收到后，需要调整自己的window size然后给C一个应答：

```
Source port: mit-dov (91)
Destination port: 54133 (54133)
[Stream index: 0]
Sequence number: 0x0f8baa6f
Acknowledgment number: 0x0b97d19d
Header length: 32 bytes
Flags: 0x010 (ACK)
Window size value: 8235
[Calculated window size: 131760]
[Window size scaling factor: 16]
Checksum: 0xcf34 [validation disabled]
Options: (12 bytes), No-Operation (NOP), No-Operation (NOP), Timestamps
```

<h2>数据传输</h2>

4个包交换完毕后，链接已经建立，然后C向S发数据："iam:a"。这个时候，PSH和ACK两个flag被置为1，PSH通知S有数据到来，需要写入。同时将数据装入TCP的data部分。由于有了data，下一条seq num的值要加上data的size（5）

```
Ack: 1, Len: 5
Source port: 54133 (54133)
Destination port: mit-dov (91)
[Stream index: 0]
Sequence number: 0x0b97d19d
[Next sequence number: 0x0b97d19d+0x5]
Acknowledgment number:0x0f8baa6f
Header length: 32 bytes
Flags: 0x018 (PSH, ACK)
Window size value: 8235
[Calculated window size: 131760]
[Window size scaling factor: 16]
Checksum: 0x6552 [validation disabled]
Options: (12 bytes), No-Operation (NOP), No-Operation (NOP), Timestamps
[SEQ/ACK analysis]
Data (5 bytes)0000  69 61 6d 3a 61                                    iam:a
Data: 69616d3a61
[Length: 5]
```

S收到后，要给出ACK应答，重点关注Ack Num：

```
Source port: mit-dov (91)
Destination port: 54133 (54133)
[Stream index: 0]
Sequence number:0x0f8baa6f
Acknowledgment number: 0x0b97d1a2
Header length: 32 bytes
Flags: 0x010 (ACK)
Window size value: 8235
[Calculated window size: 131760]
[Window size scaling factor: 16]
Checksum: 0x69bf [validation disabled]
Options: (12 bytes), No-Operation (NOP), No-Operation (NOP), Timestamps
```

随后的一个包是S向C发的[PSH,ACK]，然后C应答[ACK]。过程同上。

<h2>四次挥手</h2>

TCP结束时需要四个包，假设C向S请求关闭连接。S会向C发送[FIN,ACK]:

```
Source port: 54133 (54133)
Destination port: mit-dov (91)
[Stream index: 0]
Sequence number: 12    (relative sequence number)
Acknowledgment number: 14    (relative ack number)
Header length: 32 bytes
Flags: 0x011 (FIN, ACK)
Window size value: 8234
[Calculated window size: 131744]
[Window size scaling factor: 16]
Checksum: 0x3485 [validation disabled]
```

C收到后，会先[ACK]，然后同样发送一个[FIN,ACK]到S：

```
Source port: 54133 (54133)
Destination port: mit-dov (91)
[Stream index: 0]
Sequence number: 12    (relative sequence number)
Acknowledgment number: 14    (relative ack number)
Header length: 32 bytes
Flags: 0x011 (FIN, ACK)
Window size value: 8234
[Calculated window size: 131744]
[Window size scaling factor: 16]
Checksum: 0x3485 [validation disabled]
```

S收到后，会给C一个[ACK]，然后连接终止。