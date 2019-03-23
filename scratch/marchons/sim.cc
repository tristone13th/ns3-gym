/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Piotr Gawlowicz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 * Based on script: ./examples/tcp/tcp-variants-comparison.cc
 *
 * Topology:
 *
 *   Right Leafs (Clients)                      Left Leafs (Sinks)
 *           |            \                    /        |
 *           |             \    bottleneck    /         |
 *           |              R0--------------R1          |
 *           |             /                  \         |
 *           |   access   /                    \ access |
 *           N -----------                      --------N
 */

#include <iostream>
#include <fstream>
#include <string>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/applications-module.h"
#include "ns3/error-model.h"
#include "ns3/tcp-header.h"
#include "ns3/enum.h"
#include "ns3/event-id.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/traffic-control-module.h"

#include "ns3/opengym-module.h"
#include "tcp-rl.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpVariantsComparison");

static std::vector<uint32_t> rxPkts;

// 回调，来一个包自增
static void
CountRxPkts (uint32_t sinkId, Ptr<const Packet> packet, const Address &srcAddr)
{
  rxPkts[sinkId]++;
}

// 把服务器每个节点接收到的包数打印出来
static void
PrintRxCount ()
{
  uint32_t size = rxPkts.size ();
  NS_LOG_UNCOND ("RxPkts:");
  for (uint32_t i = 0; i < size; i++)
    {
      NS_LOG_UNCOND ("---SinkId: " << i << " RxPkts: " << rxPkts.at (i));
    }
}

// 主函数入口
int
main (int argc, char *argv[])
{

  // 设置opengym接口的端口值，通过该端口值连接到ns3env.
  uint32_t openGymPort = 5555;
  double tcpEnvTimeStep = 0.1;

  // 网络结构
  uint32_t nLeaf = 1;
  std::string transport_prot = "TcpRl";
  double error_p = 0.0;

  // 设置瓶颈链路带宽、瓶颈链路延迟、接入带宽、接入延迟
  std::string bottleneck_bandwidth = "2Mbps";
  std::string bottleneck_delay = "0.01ms";
  std::string access_bandwidth = "10Mbps";
  std::string access_delay = "20ms";
  std::string prefix_file_name = "TcpVariantsComparison";
  uint64_t data_mbytes = 0;

  // 默认MTU的字节数为400
  uint32_t mtu_bytes = 400;
  double duration = 10.0;
  uint32_t run = 0;
  bool flow_monitor = true;
  bool sack = true;
  std::string queue_disc_type = "ns3::PfifoFastQueueDisc";
  std::string recovery = "ns3::TcpClassicRecovery";

  CommandLine cmd;

  // 一些可供调用的参数设置
  cmd.AddValue ("openGymPort", "Opengym环境端口号. 默认: 5555", openGymPort);
  cmd.AddValue ("simSeed", "随机数生成器种子. 默认: 1", run);
  cmd.AddValue ("envTimeStep", "基于时间的TCP时间间隔，单位（s）. 默认: 0.1s", tcpEnvTimeStep);
  cmd.AddValue ("nLeaf", "Number of left and right side leaf nodes", nLeaf);
  cmd.AddValue ("transport_prot",
                "可供选择的协议: TcpNewReno, "
                "TcpHybla, TcpHighSpeed, TcpHtcp, TcpVegas, TcpScalable, TcpVeno, "
                "TcpBic, TcpYeah, TcpIllinois, TcpWestwood, TcpWestwoodPlus, TcpLedbat, "
                "TcpLp, TcpRl, TcpRlTimeBased",
                transport_prot);
  cmd.AddValue ("error_p", "设置环境的错包率", error_p);
  cmd.AddValue ("bottleneck_bandwidth", "瓶颈链路带宽", bottleneck_bandwidth);
  cmd.AddValue ("bottleneck_delay", "瓶颈链路延迟", bottleneck_delay);
  cmd.AddValue ("access_bandwidth", "接入链路带宽", access_bandwidth);
  cmd.AddValue ("access_delay", "接入链路延迟", access_delay);
  cmd.AddValue ("prefix_name", "Prefix of output trace file", prefix_file_name);
  cmd.AddValue ("data", "有多少兆的数据需要转发", data_mbytes);
  cmd.AddValue ("mtu", "每个数据包有多少字节", mtu_bytes);
  cmd.AddValue ("duration", "模拟的时间（秒）", duration);
  cmd.AddValue ("run", "Run index (for setting repeatable seeds)", run);
  cmd.AddValue ("flow_monitor", "启用流监控", flow_monitor);
  cmd.AddValue ("queue_disc_type", "Queue disc type for gateway (e.g. ns3::CoDelQueueDisc)",
                queue_disc_type);
  cmd.AddValue ("sack", "Enable or disable SACK option", sack);
  cmd.AddValue ("recovery", "Recovery algorithm type to use (e.g., ns3::TcpPrrRecovery", recovery);
  cmd.Parse (argc, argv);

  // 设置协议相关
  transport_prot = std::string ("ns3::") + transport_prot;

  // 设置随机数种子
  SeedManager::SetSeed (1);
  SeedManager::SetRun (run);

  // 记录日志（是否使用了TcpRl，记录端口）
  NS_LOG_UNCOND ("Ns3Env parameters:");
  if (transport_prot.compare ("ns3::TcpRl") == 0 or
      transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
    {
      NS_LOG_UNCOND ("--openGymPort: " << openGymPort);
    }
  else
    {
      NS_LOG_UNCOND ("--openGymPort: No OpenGym");
    }

  NS_LOG_UNCOND ("--seed: " << run);
  NS_LOG_UNCOND ("--Tcp version: " << transport_prot);

  // 和Agent的接口
  Ptr<OpenGymInterface> openGymInterface;
  if (transport_prot.compare ("ns3::TcpRl") == 0)
    {
      openGymInterface = OpenGymInterface::Get (openGymPort);
      Config::SetDefault ("ns3::TcpRl::Reward",
                          DoubleValue (10.0)); // Reward when increasing congestion window
      Config::SetDefault ("ns3::TcpRl::Penalty",
                          DoubleValue (-30.0)); // Penalty when decreasing congestion window
    }

  if (transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
    {
      openGymInterface = OpenGymInterface::Get (openGymPort);
      Config::SetDefault ("ns3::TcpRlTimeBased::StepTime",
                          TimeValue (Seconds (tcpEnvTimeStep))); // Time step of TCP env
    }

  // 有效载荷通过最大传输单元（MTU）减去TCP头部与IP头部
  NS_LOG_UNCOND ("MTU is: " << mtu_bytes);
  Header *temp_header = new Ipv4Header ();
  uint32_t ip_header = temp_header->GetSerializedSize ();
  NS_LOG_UNCOND ("IP Header size is: " << ip_header);
  delete temp_header;
  temp_header = new TcpHeader ();
  uint32_t tcp_header = temp_header->GetSerializedSize ();
  NS_LOG_UNCOND ("TCP Header size is: " << tcp_header);
  delete temp_header;
  uint32_t tcp_adu_size = mtu_bytes - 20 - (ip_header + tcp_header);
  NS_LOG_UNCOND ("TCP ADU size is: " << tcp_adu_size);

  // 模拟开始与结束的时间
  double start_time = 0.1;
  double stop_time = start_time + duration;

  // 4 MB of TCP buffer
  // 发送的缓冲区大小是2M，接收的缓冲区大小是2M
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (1 << 21));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (1 << 21));

  // SACK(Selective ACK)是TCP选项，它使得接收方能告诉发送方哪些报文段丢失，哪些报文段重传了，哪些报文段已经提前收到等信息。根据这些信息TCP就可以只重传哪些真正丢失的报文段。
  Config::SetDefault ("ns3::TcpSocketBase::Sack", BooleanValue (sack));

  // set ns3::TcpSocket::DelAckCount to 0, to disable delayed ACKs.
  Config::SetDefault ("ns3::TcpSocket::DelAckCount", UintegerValue (2));

  Config::SetDefault ("ns3::TcpL4Protocol::RecoveryType",
                      TypeIdValue (TypeId::LookupByName (recovery)));
  // Select TCP variant
  if (transport_prot.compare ("ns3::TcpWestwoodPlus") == 0)
    {
      // TcpWestwoodPlus is not an actual TypeId name; we need TcpWestwood here
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TcpWestwood::GetTypeId ()));
      // the default protocol type in ns3::TcpWestwood is WESTWOOD
      Config::SetDefault ("ns3::TcpWestwood::ProtocolType", EnumValue (TcpWestwood::WESTWOODPLUS));
    }
  else
    {
      TypeId tcpTid;
      NS_ABORT_MSG_UNLESS (TypeId::LookupByNameFailSafe (transport_prot, &tcpTid),
                           "TypeId " << transport_prot << " not found");
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TypeId::LookupByName (transport_prot)));
    }

  // 错误率模型，设置一次直接损失一个packet，错误率采用Error_p，随机变量采用UniformRandomVariable
  Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable> ();
  uv->SetStream (50);
  RateErrorModel error_model;
  error_model.SetRandomVariable (uv);
  error_model.SetUnit (RateErrorModel::ERROR_UNIT_PACKET);
  error_model.SetRate (error_p);

  // 网络结构的创建
  // 点到点模型
  PointToPointHelper bottleNeckLink;
  bottleNeckLink.SetDeviceAttribute ("DataRate", StringValue (bottleneck_bandwidth));
  bottleNeckLink.SetChannelAttribute ("Delay", StringValue (bottleneck_delay));
  bottleNeckLink.SetDeviceAttribute ("ReceiveErrorModel", PointerValue (&error_model));

  PointToPointHelper pointToPointLeaf;
  pointToPointLeaf.SetDeviceAttribute ("DataRate", StringValue (access_bandwidth));
  pointToPointLeaf.SetChannelAttribute ("Delay", StringValue (access_delay));

  // 使用点到点链路创建一个哑铃拓扑结构的模型，其实只需要传入链路作为参数即可，节点将会自动进行创建
  PointToPointDumbbellHelper d (nLeaf, pointToPointLeaf, nLeaf, pointToPointLeaf, bottleNeckLink);

  // 给所有的节点安装IP协议
  InternetStackHelper stack;
  stack.InstallAll ();

  // 传输控制（路由器的排队规则）
  TrafficControlHelper tchPfifo;
  tchPfifo.SetRootQueueDisc ("ns3::PfifoFastQueueDisc");

  TrafficControlHelper tchCoDel;
  tchCoDel.SetRootQueueDisc ("ns3::CoDelQueueDisc");

  DataRate access_b (access_bandwidth);
  DataRate bottle_b (bottleneck_bandwidth);
  Time access_d (access_delay);
  Time bottle_d (bottleneck_delay);

  uint32_t size = static_cast<uint32_t> ((std::min (access_b, bottle_b).GetBitRate () / 8) *
                                         ((access_d + bottle_d + access_d) * 2).GetSeconds ());

  // 设置路由器中队列的大小
  Config::SetDefault ("ns3::PfifoFastQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, size / mtu_bytes)));
  Config::SetDefault ("ns3::CoDelQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::BYTES, size)));

  // 在左右两个路由器中根据设置进行相应的排队规则的安装
  if (queue_disc_type.compare ("ns3::PfifoFastQueueDisc") == 0)
    {
      tchPfifo.Install (d.GetLeft ()->GetDevice (1));
      tchPfifo.Install (d.GetRight ()->GetDevice (1));
    }
  else if (queue_disc_type.compare ("ns3::CoDelQueueDisc") == 0)
    {
      tchCoDel.Install (d.GetLeft ()->GetDevice (1));
      tchCoDel.Install (d.GetRight ()->GetDevice (1));
    }
  else
    {
      NS_FATAL_ERROR ("Queue not recognized. Allowed values are ns3::CoDelQueueDisc or "
                      "ns3::PfifoFastQueueDisc");
    }

  // 分配IP地址
  // 左边的所有叶子节点的地址为10.1.1.x，子网掩码为255.255.255.0
  // 右边的所有叶子节点的地址为10.2.1.x，子网掩码为255.255.255.0
  // 路由器的接口地址为10.3.1.x，子网掩码为255.255.255.0
  d.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.1.0", "255.255.255.0"),
                         Ipv4AddressHelper ("10.2.1.0", "255.255.255.0"),
                         Ipv4AddressHelper ("10.3.1.0", "255.255.255.0"));

  NS_LOG_INFO ("Initialize Global Routing.");

  // Brief Build a routing database and initialize the routing tables of the nodes in the simulation.
  // Makes all nodes in the simulation into routers.
  // 简单地构建一个路由数据库并且初始化一个关于模拟中节点的路由表，让路由器感知到所有的节点。
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // 在左右两边安装应用，所有节点的所有应用的端口都是50000
  uint16_t port = 50000;
  Address sinkLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
  PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory", sinkLocalAddress);
  ApplicationContainer sinkApps;
  for (uint32_t i = 0; i < d.RightCount (); ++i)
    {
      sinkHelper.SetAttribute ("Protocol", TypeIdValue (TcpSocketFactory::GetTypeId ()));
      sinkApps.Add (sinkHelper.Install (d.GetRight (i)));
    }
  sinkApps.Start (Seconds (0.0));
  sinkApps.Stop (Seconds (stop_time));

  for (uint32_t i = 0; i < d.LeftCount (); ++i)
    {
      // BulkSendHelper是一个极尽所能来发送流量的类，它试图填满当前的带宽
      AddressValue remoteAddress (InetSocketAddress (d.GetRightIpv4Address (i), port));
      Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (tcp_adu_size));
      BulkSendHelper ftp ("ns3::TcpSocketFactory", Address ());
      ftp.SetAttribute ("Remote", remoteAddress);
      ftp.SetAttribute ("SendSize", UintegerValue (tcp_adu_size));
      ftp.SetAttribute ("MaxBytes", UintegerValue (data_mbytes * 1000000));

      ApplicationContainer clientApp = ftp.Install (d.GetLeft (i));
      clientApp.Start (Seconds (start_time * i)); // Start after sink
      clientApp.Stop (Seconds (stop_time - 3)); // Stop before the sink
    }

  // 流监控，在所有的节点上安装flowhelper流监控
  FlowMonitorHelper flowHelper;
  if (flow_monitor)
    {
      flowHelper.InstallAll ();
    }

  // 初始化右边每一个节点接收到包的数量
  for (uint32_t i = 0; i < d.RightCount (); ++i)
    {
      rxPkts.push_back (0);
      Ptr<PacketSink> pktSink = DynamicCast<PacketSink> (sinkApps.Get (i));
      pktSink->TraceConnectWithoutContext ("Rx", MakeBoundCallback (&CountRxPkts, i));
    }

  // 设置结束的时间
  Simulator::Stop (Seconds (stop_time));

  // 开始运行
  Simulator::Run ();

  // 保存流监控记录
  if (flow_monitor)
    {
      flowHelper.SerializeToXmlFile (prefix_file_name + ".flowmonitor", true, true);
    }

  // 如果使用了强化学习，通知结束
  if (transport_prot.compare ("ns3::TcpRl") == 0 or
      transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
    {
      openGymInterface->NotifySimulationEnd ();
    }

  // 输出右端节点每一个接收到的包数
  PrintRxCount ();

  // 变量清理等工作
  Simulator::Destroy ();
  return 0;
}
