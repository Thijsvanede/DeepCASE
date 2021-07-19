# Dataset info & mapping
In this directory we provide a full list of all events that were included in the Dataset, and provide the event mappings to expert rules for both the AlienVault and Sigma rules.

## Overview
This directory contains the following:
 * `events.txt`, a `.txt` file listing all events. This full list is also shown in the Section [Events](#events) below.
 * `mappings`, a directory containing `.json` files for both the AlienVault and Sigma rules and `.csv` files containing the mappings from AlienVault and Sigma events to our Lastline events.

## Events
Below, we list all events that occured in our dataset.

|Events A-P                                        |Events P-Z                                        |
|--------------------------------------------------|--------------------------------------------------|
|Active Directory Account Enumeration              |Phishing: Financial Sector                        |
|Active Directory Password Policy Query            |Phishing: Payment Service                         |
|Active Directory Trust Enumeration                |Phishing: Social Networking                       |
|Administrator RDP Connection                      |Phishing: e-Commerce                              |
|Adwind RAT                                        |PoisonIvy                                         |
|Ambiguous Transfer-Encoding header field          |Pony                                              |
|Ammyy Admin Check-In                              |Port Scan                                         |
|Andromeda                                         |Port Sweep                                        |
|Anomalous Ammyy Admin Check-In                    |Possible Apache Struts OGNL Exploit Attempt       |
|Anomalous Command Execution Over SMB              |Possible Cross Site Scripting Attempt             |
|Anomalous connection between two hosts            |Possible Nyetya Lateral Movement                  |
|Babylon                                           |Potential AD Account Enumeration                  |
|Bandook                                           |Potential Apache Struts Exploitation Attempt      |
|Beaconing activity                                |Potential CVE-2019-9978 Exploitation Attempt      |
|Bedep                                             |Potential FTP URL stream injection                |
|BetaBot                                           |Potential Lateral Movement: Psexec SMB Create     |
|BitCoin Miner                                     |Potential Nyetya Lateral Movement                 |
|BitCoinMiner                                      |Potential Remote Shell Command                    |
|BitTorrent                                        |Potential SMB Brute Force Attack                  |
|BlackHole                                         |Potential SMB probe for MS17-010 patch            |
|Bladabindi                                        |Potential ThinkPHP RCE Exploitation Attempt       |
|Blocked HTTP Connection                           |Potential Web Shell Command                       |
|BrowseFox                                         |PsExec interaction                                |
|Buran                                             |Pykspa                                            |
|CNCERT CC Sinkhole                                |Qakbot                                            |
|CVE-2010-3275                                     |Qarallax                                          |
|CVE-2011-2371                                     |Qualys Vulnerability Scan                         |
|CVE-2017-7269 Exploit                             |RDP Connection with Nessus Cookie                 |
|CVE-2018-13379 Exploit                            |RDP Connection with nmap Cookie                   |
|CVE-2018-2628 Exploit                             |RDP Connection with rapid Cookie                  |
|CVE-2018-7600 Exploit                             |RDP Jump Box                                      |
|CVE-2019-0708                                     |RDP protocol                                      |
|CVE-2019-1003000 Exploit                          |RIG Exploit kit                                   |
|CVE-2019-16759 Exploit                            |Ramnit                                            |
|CVE-2019-19781 Exploit                            |Ranbyus                                           |
|CVE-2020-0601 Exploit                             |Recently registered domain access                 |
|CVE-2020-0796 Exploit                             |Recslurp                                          |
|Canadian Pharmacy Online Drugstore Scam           |RelevantKnowledge                                 |
|Carberp                                           |Remcos RAT                                        |
|Certificate with Explicit EC Parameters           |Remote Task Scheduling                            |
|China Chopper Web Shell                           |Revenge RAT                                       |
|CoinHive Monero Miner                             |SMB2 Create for an .exe file                      |
|CoinMiner                                         |SMBv1                                             |
|Command Execution Over SMB                        |SMTP Clear-Text Password Transmission             |
|Conficker                                         |SQL Injection                                     |
|Connection on suspicious port                     |SSH connection on unusual port                    |
|Connection on unusual port                        |Sality                                            |
|Connection received on unusual port               |Self-signed certificate on high port              |
|Consecutive failed SMB login attempts             |Shellshock Exploit Attempt                        |
|Consecutive failed Telnet logins                  |Shlayer                                           |
|Crimson RAT                                       |Simda                                             |
|CryptoWall                                        |Sinkdns Sinkhole                                  |
|Cryptolocker                                      |Sinkhole host                                     |
|DGA activity                                      |Spigot                                            |
|DNS Tunneling                                     |Spyrat                                            |
|DNS Zone Transfer Query                           |SuppoBox                                          |
|DNS-over-HTTPS                                    |Suspicious Account Enumeration                    |
|Dark Hydrus                                       |Suspicious DNS Resolution                         |
|Data download on suspicious port                  |Suspicious DNS Zone Transfer                      |
|Data upload on suspicious port                    |Suspicious Dynamic DNS Public IP Check            |
|Dealply                                           |Suspicious FTP Upload to an External Server       |
|Delivery errors for outbound SMTP                 |Suspicious Kerberos AS_REQ RC4 Encryption         |
|Deprecated SSL Version Usage                      |Suspicious Kerberos Authentication Failure        |
|Dinwod                                            |Suspicious LDAP Authentication Failure            |
|Domain involved in spam emails                    |Suspicious Password Policy Query                  |
|Download Of File With Malicious PowerShell        |Suspicious RDP Jump Box                           |
|Dridex                                            |Suspicious RDP connection                         |
|Dropper attempt to download payload               |Suspicious Redirection Gateway Domain             |
|Drupalgeddon 2                                    |Suspicious Remote Shell Command                   |
|Dyreza RAT                                        |Suspicious Remote Task Scheduling                 |
|EXE Embedded in page                              |Suspicious SMB Permission Errors                  |
|Emotet                                            |Suspicious SSH connection                         |
|Empire Listener                                   |Suspicious SSL certificate                        |
|External IP Lookup                                |Suspicious TLS Certificate                        |
|FTP Clear-Text Password Transmission              |Suspicious TLS Certificate Fields                 |
|FTP File Upload to a Public Server                |Suspicious Tor Connection                         |
|FTP Telnet Evasion                                |Suspicious Trusted Domain Enumeration             |
|Fake Virus Phone Scam                             |Suspicious URL                                    |
|Fareit                                            |Suspicious Use of DNS-Over-HTTPS                  |
|Farfli                                            |Suspicious VNC Connection                         |
|Feodo                                             |Suspicious VPN connection                         |
|FileBulldog                                       |Suspicious Wordpress URL                          |
|FlyStudio                                         |Suspicious data download                          |
|Formbook                                          |Suspicious data upload                            |
|Fynloski                                          |Suspicious javascript obfuscation                 |
|Gamarue                                           |Suspicious use of SMBv1                           |
|Gaming Client                                     |Suspicious use of self-signed certificate         |
|Generic FakeAV                                    |TDLClickServer                                    |
|GenericClickFraud                                 |TDS_Redirect                                      |
|GenericTrojan_11                                  |TELNET Clear-Text Password Transmission           |
|GomyHit                                           |TLS Certificate With Default Values               |
|HTTP 1xx Response With Body Content               |Time Series Anomaly                               |
|HTTP 414 URI Too Long                             |Tofsee                                            |
|HTTP Basic Authentication                         |Tor                                               |
|Hupigon                                           |Transfer-Encoding in HTTP 1.0                     |
|IMAP Clear-Text Password Transmission             |TrickBot                                          |
|IcedID                                            |URLhaus blacklisted host                          |
|Injected Coin Miner in Compromised Website        |USPSDapato                                        |
|Injected redirection to suspicious site           |Uncommon HTTP 1xx Response With Body Content      |
|InstallCore                                       |Unknown Crypto Miner                              |
|Kerberos AS-REQ with RC4 Encryption               |Unusual Access to Recently Registered Domain      |
|Kerberos Authentication Failure                   |Unusual Beaconing Activity                        |
|Khelios                                           |Unusual DGA Activity                              |
|Kovter                                            |Unusual DNS Resolution                            |
|LDAP Authentication Failure                       |Unusual DNS Tunneling                             |
|Lastline blocking test                            |Unusual External IP Lookup                        |
|Lastline sensor rule test                         |Unusual HTTP Basic Authentication                 |
|Lastline test                                     |Unusual JA3 fingerprint observed                  |
|Lethic                                            |Unusual Port Scan                                 |
|Linkury                                           |Unusual Port Sweep                                |
|Locky                                             |Unusual Possible Cross Site Scripting Attempt     |
|Login to cryptocurrency mining pool               |Unusual Potential Apache Struts Exploitation      |
|Login to pastebin.com API                         |Unusual SMB2 Create for an .exe file              |
|Loki Bot                                          |Unusual connection between two hosts              |
|Luminosity                                        |Unusual connection received on unusual port       |
|MS Sinkhole Resolved                              |Unusual data download                             |
|Magecart                                          |Unusual data upload                               |
|Malicious Binary Download                         |Unusual delivery errors for outbound SMTP         |
|Malicious Document Download                       |Unusual user agent string observed                |
|Malicious File Download                           |Upatre                                            |
|Malicious Script Download                         |Ursnif                                            |
|Malicious android app download                    |VBS.Jenxcus                                       |
|Malicious archive download                        |VNC Traffic                                       |
|Malicious java app download                       |VPN Traffic                                       |
|Malicious redirector                              |Virut                                             |
|Metasploit WinRM                                  |WSHRAT                                            |
|Metasploit web server                             |Web Application Attack                            |
|Microsoft Watson                                  |Web Shell Command                                 |
|Mirai Login Attempt                               |WinHTTP User-Agent                                |
|Mirai Variant                                     |WinWrapper                                        |
|NanoCore                                          |Winnti                                            |
|Necurs                                            |Winnti Scanner                                    |
|Nemty                                             |XMR-Stak                                          |
|Nessus                                            |XMRig                                             |
|NetwiredRC                                        |Xtrat                                             |
|Network Defender rule match                       |Yoddos                                            |
|Neutrino                                          |Zegost                                            |
|Nitol                                             |ZeroAccess                                        |
|Nmap                                              |ZeuS Gameover                                     |
|Obfuscated JavaScript                             |Zusy                                              |
|OpenVAS                                           |http-evader Test Suite                            |
|OxyPumper                                         |libtorrent                                        |
|POP3 Clear-Text Password Transmission             |uTorrent                                          |
|PhishTank blacklisted host                        |unescape of long unicode string                   |
|Phishing                                          |windows-x64-vncinject-reverse_tcp_uuid            |
