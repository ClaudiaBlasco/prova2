#!/usr/bin/env python3
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node
from mininet.log import setLogLevel, info
from mininet.cli import CLI

class LinuxRouter(Node):
    """Node amb IP forwarding habilitat"""
    def config(self, **params):
        super(LinuxRouter, self).config(**params)
        self.cmd('sysctl -w net.ipv4.ip_forward=1')

    def terminate(self):
        self.cmd('sysctl -w net.ipv4.ip_forward=0')
        super(LinuxRouter, self).terminate()

class EmpresaTopo(Topo):
    """Topologia ampliada amb xarxes IT, Marketing, HR i Servidors"""

    def build(self):

        # ============================
        # 1) CREAR ROUTER
        # ============================
        r0 = self.addNode('r0', cls=LinuxRouter, ip='192.168.1.1/24')

        # ============================
        # 2) CREAR SWITCHES
        # ============================
        s1 = self.addSwitch('s1')   # IT
        s2 = self.addSwitch('s2')   # Marketing
        s3 = self.addSwitch('s3')   # HR
        s4 = self.addSwitch('s4')   # Servidors

        # ============================
        # 3) ENLLAÇAR SWITCHES AL ROUTER
        # ============================
        self.addLink(s1, r0, intfName2='r0-eth1', params2={'ip': '192.168.1.1/24'})
        self.addLink(s2, r0, intfName2='r0-eth2', params2={'ip': '172.16.0.1/12'})
        self.addLink(s3, r0, intfName2='r0-eth3', params2={'ip': '10.0.0.1/8'})
        self.addLink(s4, r0, intfName2='r0-eth4', params2={'ip': '192.168.100.1/24'})

        # ============================
        # 4) CREAR HOSTS
        # ============================

        # Xarxa IT
        h1 = self.addHost('h1', ip='192.168.1.100/24', defaultRoute='via 192.168.1.1')
        h2 = self.addHost('h2', ip='192.168.1.101/24', defaultRoute='via 192.168.1.1')

        # Xarxa Marketing
        h3 = self.addHost('h3', ip='172.16.0.100/12', defaultRoute='via 172.16.0.1')
        h4 = self.addHost('h4', ip='172.16.0.101/12', defaultRoute='via 172.16.0.1')

        # Xarxa HR
        h5 = self.addHost('h5', ip='10.0.0.100/8', defaultRoute='via 10.0.0.1')
        h6 = self.addHost('h6', ip='10.0.0.101/8', defaultRoute='via 10.0.0.1')

        # Xarxa Servidors
        web = self.addHost('webserver',  ip='192.168.100.10/24', defaultRoute='via 192.168.100.1')
        db  = self.addHost('dbserver',   ip='192.168.100.20/24', defaultRoute='via 192.168.100.1')
        files = self.addHost('fileserver', ip='192.168.100.30/24', defaultRoute='via 192.168.100.1')

        # ============================
        # 5) ENLLAÇAR HOSTS ALS SWITCHES
        # ============================
        self.addLink(h1, s1)
        self.addLink(h2, s1)

        self.addLink(h3, s2)
        self.addLink(h4, s2)

        self.addLink(h5, s3)
        self.addLink(h6, s3)

        self.addLink(web, s4)
        self.addLink(db, s4)
        self.addLink(files, s4)

def run():
    topo = EmpresaTopo()
    net = Mininet(topo=topo, waitConnected=True)
    net.start()

    r0 = net['r0']
    info("\n### Configurant FIREWALL ###\n")

    # ============================================================
    # FIREWALL IPTABLES SEGONS LES RESTRICCIONS
    # ============================================================

    # a) Xarxa IT (192.168.1.0/24) → ACCÉS COMPLET als servidors
    r0.cmd("iptables -A FORWARD -s 192.168.1.0/24 -d 192.168.100.0/24 -j ACCEPT")

    # b) Xarxa HR (10.0.0.0/8) → NOMÉS SERVIDOR WEB
    r0.cmd("iptables -A FORWARD -s 10.0.0.0/8 -d 192.168.100.10 -j ACCEPT")   # OK WEB
    r0.cmd("iptables -A FORWARD -s 10.0.0.0/8 -d 192.168.100.20 -j DROP")     # NO DB
    r0.cmd("iptables -A FORWARD -s 10.0.0.0/8 -d 192.168.100.30 -j DROP")     # NO FILES

    # c) Xarxa Marketing (172.16.0.0/12) → NO ACCÉS SERVIDORS
    r0.cmd("iptables -A FORWARD -s 172.16.0.0/12 -d 192.168.100.0/24 -j DROP")

    # ============================================================
    # CONFIGURAR SERVIDOR WEB
    # ============================================================
    web = net['webserver']
    web.cmd('echo "<h1>Servidor Web Corporatiu</h1>" > /tmp/index.html')
    web.cmd('python3 -m http.server 80 --directory /tmp/ > /dev/null 2>&1 &')

    info("\n### Firewalls i servidor web configurats! ###\n")
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()
