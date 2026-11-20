// Exercise the production stage up to the shared quantization entry.
#include <cassert>
#include <string>

constexpr int AIV = 1;
enum class HierarchicalSignal { Sequence, LocalRailPacked };
struct AscendTimer {
    enum Event { AIV_INPUT_BUFFER_INIT, AIV_QUANT };
    void Start(Event) {}
    void End(Event) {}
    bool Enabled()
    {
        return false;
    }
};
template <int g_coreType>
struct Stage {
    struct {
        bool tilingData;
        const void* inputScaleGmAddr;
    } params_;
    struct {
        int jobIndex;
    } aivJob_;
    AscendTimer timer_;
    std::string events;
    int commonConfig_{}, g_winRankAddr_{}, sendMaskConfig_{}, sendMaskScratch_{}, rankId_{};
    bool UseHierarchicalRoce(bool enabled)
    {
        return enabled;
    }
    int HierarchicalPackJob()
    {
        return 1;
    }
    int HierarchicalSendJob()
    {
        return 2;
    }
    int HierarchicalHost(int)
    {
        return 0;
    }
    void InitRailIdsScratch()
    {
        events += 'U';
    }
    void SendAndQuantBuffInit()
    {
        events += 'F';
    }
    template <class T>
    void PublishHierarchicalLocalSignal(T&, HierarchicalSignal s)
    {
        events += s == HierarchicalSignal::Sequence ? 'S' : 'L';
    }
    template <class... T>
    void PackHierarchicalCombineWeights(T&&...)
    {
        events += 'W';
    }
    template <class... T>
    void SendTopkIdsToRailServer(T&&...)
    {
        events += 'I';
    }
    template <class... T>
    void PostHierarchicalDispatch(T&&...)
    {
        events += 'P';
    }
    void PackOrPostRailMetadata()
    { /* PACK */
    }
    void ProcessInputPreparationStage()
    { /* STAGE */
    }
};

template <int core>
void check()
{
#ifdef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
    constexpr bool hns = true;
#else
    constexpr bool hns = false;
#endif
    for (bool hierarchical : {false, true}) {
        for (bool fp8 : {false, true}) {
            for (int role : {0, 1, 2}) {
                Stage<core> stage{};
                stage.params_ = {hierarchical, fp8 ? &stage : nullptr};
                stage.aivJob_.jobIndex = role;
                stage.ProcessInputPreparationStage();
                std::string expected = "F";
                if (core == AIV && hierarchical) {
                    const bool early = hns && !fp8;
                    if (role == 1) expected = early ? "USWILF" : "FSWIL";
                    if (role == 2) expected = early ? "PF" : "FP";
                }
                assert(stage.events == expected);
            }
        }
    }
}
int main()
{
    check<AIV>();
    check<0>();
}
