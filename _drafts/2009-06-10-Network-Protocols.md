## HTTPs

### HTTPs

- HTTPs:Hypertext Transfer Protocol over Secure Socket Layer.位于应用层的基于SSL/TLS的HTTP协议

- SSL/TLS
	- 本身是带有加密信息的传输层协议
	- SSL：为网络通信提供安全及数据完整性的安全协议
	- TLS：基于SSL之上的通用协议，为SSL的继任者

- SSL加密方式
- HTTPs建立链接的步骤:

	- Computers agree on how to encrypt
		- C => S ： 客户端向server发送一个hello包，提供几种加密方式:

			- Key：RSA,Diffle-Hellman,DSA

			- Cipher：AES, RC4, Triple DES

			- Hash：HMAC-MD5

		- S => C : 通知客户端，server选取了哪几种机密方式

	- Server sends a certificate to client

	- Client says 'start encrypting'

		- Client Key Exchange : 双方计算master secret code, 这个secret code用来做后续的加密工作

		- change cipher spec：让server选择上一个步骤中约定的加密方式

	- The server says 'start encrypting'
