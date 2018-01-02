- 包中的内容
	- armv7
	- armv7s
	- arm64
	- 图片
	- 其它各种资源

### On Demand Resources

- Asset packs are built by Xcode
- Can contain any non-executable assets
- Hosted by the App Store
- Downloaded as appropriate
- Device-thinned just like the other content

### Advantages

- Support devices with constrained storage
- Shorter download times
- Easier to stay with over-the-air size limits
- Support more types of devices with less compromise

### Asset Slicing

- Asset Catalogs
	- 根据不同的设备组织资源
	- App Thin必须要App中使用asset catelog

### Device Traits

- Devices hava a key set of characteristics which assets can be optimized for

	- Screen resolution
	- Device family

iOS9新增加了两种traits:

- Graphics capabilities
	- Meta GPUFamily1, Metal GPU Family2

- Memory Level
	- 1GB, 2GB

### Asset Catalogs包含

- Named Images： 程序中的切图，会被压缩

- Named Data: iOS9中新增的分类：存放任意类型的问题，    
