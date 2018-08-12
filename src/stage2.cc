#include "stage2.h"
#include "parse.h"
#include "hash.h"

#include <regex>

using namespace Eigen;

extern volatile bool interrupted;

template <class T, class S> Stage2<T, S>::Stage2(
    Buffer<T>& ibuffer, Buffer<S>& obuffer,
    const Options& opts
)
:
    ibuffer_(ibuffer),
    obuffer_(obuffer),
    opts(opts),
    console(spdlog::get("console"))
{
    std::string line;
    std::istringstream iss(opts.interactions);
    while(std::getline(iss, line, ',')) {
        console->info("Setting up interaction [{0}]", line);
        interact.push_back(std::vector<std::string>());
        std::string s;
        std::istringstream vis(line);
        while(std::getline(vis, s, '*')) {
            console->debug("... namespace [{0}]", s);
            interact.back().push_back(s);
        }
    }
}

template <class T, class S>
void Stage2<T, S>::compile(std::regex& regex, const std::string& sregex, const std::string& name) {
    if(sregex.empty())
        return;
    try {
        regex.assign(sregex);
        console->debug("{0} [{0}].", name, sregex);
    } catch (std::regex_error& e) {
        console->error("Invalid regex [{0}].", opts.relogit);
    }
}

template <class T, class S>
void Stage2<T, S>::run()
{
    murmur3<> h(static_cast<size_t>(opts.seed));
    size_t lineno = 0;
    std::istringstream iss;
    std::istringstream issv;

    std::vector<std::pair<size_t, double>> fids;
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, double>>>> fns;
    std::array<std::vector<std::pair<std::string,double>>, 2ul> ifs;
    std::unordered_map<size_t, std::string> nmap;
    std::vector<std::pair<double, std::vector<std::pair<size_t, double>>>> batchbuf;
    std::vector<std::string> descbuf;

    std::regex relogit, relog, redesc, reignore, rekeep;

    compile(relogit,  opts.relogit,  "Logit feature transformation for");
    compile(relog,    opts.relog,    "Log feature transformation for");
    compile(redesc,   opts.desc,     "Description using");
    compile(reignore, opts.reignore, "Ignoring features");
    compile(rekeep,   opts.rekeep,   "Keeping features");

    while (true) {
        T line = ibuffer_.remove();
        if(!line.empty()) {
            int size;

            const int SIZE(line.size()+1);
            char* buff = new char[SIZE];
            memset(buff, 0, SIZE);


            std::copy(line.begin(), line.end(), buff);

            char** strs = splitZeroCopyInPlace(buff, ' ', &size);
            console->trace("In-place string split size = {0}", size);

            iss.clear();

            bool nolabel = true;
            bool needlabel = true;
            std::string ns("");
            double label = 0.0;

            console->trace("Parsing example line #{0}", lineno);

            std::string fn;;
            std::ostringstream ssdesc;
            size_t ufeats = 0ul;

            fns.clear();
            for(int i = 0; i < size; ++i) {
                char * s = strs[i];
                console->trace("Parsing segment #{0} {1}", i, s);

                if(s[0] != '|') {
                    if(nolabel) {
                        issv.str(s);
                        issv.clear();
                        if(issv >> label){
                            if(label < 0.0) label = 0.0; 
                            console->debug("Label [{0}]", label);
                        }
                        else if(needlabel){
                            console->error("Error reading label. Skipping line #{0}.", lineno);
                            break;
                        }
                    } else {
                        //FEATURE
                        int sz;
                        char** fsnv = splitZeroCopyInPlace(s, ':', &sz);
                        console->trace("In-place string split size = {0}: {1}", sz, fsnv[0]);
                        double fv = 1.0;
                        if(sz > 1) {
                            fv = parse(fsnv[1]);
                        }
                        std::string fn(ns);
                        fn += "^";
                        fn += fsnv[0];
                        ++ufeats;

                        //TODO basic feature transformations on demand?
                        if(!opts.desc.empty()) {
                            if (std::regex_search(fn, redesc)) {
                                ssdesc << fsnv[0];
                                if(fv != 1)
                                    ssdesc << "=" << fv;
                                ssdesc << " ";
                            }

                        }
                        if(*fsnv[0] == '\0') {
                            free(fsnv);
                            continue;
                        }

                        free(fsnv);

                        if(!opts.reignore.empty()) {
                            if (std::regex_search(fn, reignore)) {
                                continue;
                            }

                        }
                        if(!opts.rekeep.empty()) {
                            if (!std::regex_search(fn, rekeep)) {
                                continue;
                            }

                        }
                        if(!opts.relogit.empty()) {
                            //console->debug("logit regex check [{0}] ~= [{1}]", logit_regex, fn);
                            if (std::regex_search(fn, relogit)) {
                                //console->debug("applying logit");
                                const double eps = 1e-15;
                                fv = std::max(eps, std::min(1.0-eps, fv));
                                fv = std::log((1.0-fv)/fv);
                            }
                        }
                        if(!opts.relog.empty()) {
                            if (std::regex_search(fn, relog)) {
                                fv = std::log(std::max(1e-20, fv));
                            }
                        }

                        fns.back().second.push_back(std::pair<std::string, double>(fn, fv));
                    }
                } else {
                    //NAMESPACE
                    char dummy;
                    std::istringstream issv(s);
                    issv >> dummy;
                    if(issv >> ns){
                        console->trace("Namespace [{0}]", ns);
                    }
                    else{
                        console->debug("Anonymous namespace on line #{0}.", lineno);
                        ns.clear();
                    }
                    if(!opts.desc.empty()) {
                        if (std::regex_search(ns + "^",  redesc)) {
                            ssdesc << "|" <<  ns << " ";
                        }
                    }

                    fns.push_back(std::pair<std::string, std::vector<std::pair<std::string, double>>>(ns, std::vector<std::pair<std::string, double>>()));
                }
                nolabel = false;
            }
            free(strs);
            if(needlabel && nolabel)
                continue;

            //INTERACTIONS
            std::vector<std::pair<size_t, std::vector<size_t>>> interactids;

            size_t nfeats = 0ul;

            for(auto iit = interact.begin(), iiend = interact.end(); iit != iiend; ++iit) {
                auto& grp = *iit;
                std::vector<size_t> idx;
                bool fbreak = false;
                size_t nfadd = 1ul;
                for(auto ig = grp.begin(), igend = grp.end(); ig != igend; ++ig) {
                    idx.push_back(std::find_if(fns.begin(), fns.end(),  [&ig](std::pair<std::string, std::vector<std::pair<std::string, double>>>& x)->bool { return x.first == *ig; })-fns.begin());
                    nfadd *= fns[idx.back()].second.size();
                    if(idx.back() >= fns.size()) {
                        fbreak = true;
                        break;
                    }
                }
                if(fbreak) continue;
                nfeats += nfadd;
                interactids.push_back(std::pair<size_t, std::vector<size_t>>(nfadd, idx));
            }
            fids.clear();
            fids.reserve(ufeats + nfeats+1);

            //UNIGRAM
            console->debug("Adding intercept feature 0");
            fids.push_back(std::pair<size_t, double>(0, 1.0));
            if(opts.explain) {
                nmap[0ul] = std::string("intercept");
            }

            console->debug("Adding unigram features line #{0}", lineno);

            for(auto nit = fns.begin(), nend = fns.end(); nit != nend; ++nit) {
                auto& ns(nit->first);
                auto& feats(nit->second);
                console->debug("Adding {2} [{1}] namespace unigram features line #{0}", lineno, ns, feats.size());

                for(auto fit = feats.begin(), fend = feats.end(); fit != fend; ++fit) {
                    auto& fn(fit->first);
                    auto& fv(fit->second);


                    size_t fid = h(fn) % opts.N;
                    console->debug("Feature [{0}:{1}:{2}]", fn, fid, fv);
                    fids.push_back(std::pair<size_t, double>(fid, fv));
                    if(opts.explain) {
                        nmap[fid] = fn;
                    }
                }
            }

            console->debug("Adding feature interactions line #{0}", lineno);

            console->debug("Number of interaction features = {0}", nfeats);

            size_t level = 0;
            size_t cidx = 0ul;

            const std::string smult("*"), sns("^");
            ifs[0].clear();
            ifs[1].clear();
            ifs[0].reserve(nfeats);
            ifs[1].reserve(nfeats);
            for(auto iit = interactids.begin(), iiend = interactids.end(); iit != iiend; ++iit, ++level) {
                ifs[cidx].clear();
                ifs[cidx].push_back(std::pair<std::string, double>("", 1.0));
                auto& grp(*iit);
                for(auto ig = grp.second.begin(), igend = grp.second.end(); ig != igend; ++ig) {
                    const size_t oidx = 1 - cidx;
                    ifs[oidx].clear();
                    auto& ns(fns[*ig]);
                    console->debug("...namespace [{0}:{1}]", ns.first, *ig);
                    if(ns.second.size() <= 0)
                        break;
                    for(auto nit = ns.second.begin(), nend = ns.second.end(); nit != nend; ++nit) {
                        auto& fn(nit->first);
                        auto& fv(nit->second);
                        std::string fname(fn);
                        fname += smult;
                        for(auto pit = ifs[cidx].begin(), pend = ifs[cidx].end(); pit != pend; ++pit) {
                            // skip duplicate permutations of same namespace n-ary features
                            if(pit->first.find(fns[*ig].first)== 0 && pit->first.compare(fn) >= 0)
                                continue;
                            ifs[oidx].push_back(std::pair<std::string, double>(fname + pit->first, pit->second * fv));
                        }
                    }
                    cidx = oidx;
                }
                for(auto fit = ifs[cidx].begin(), fend = ifs[cidx].end(); fit != fend; ++fit) {
                    auto& fn(fit->first);
                    auto& fv(fit->second);
                    size_t fid = h(fn) % opts.N;
                    console->trace("Feature [{0}:{1}:{2}]", fn, fid, fv);
                    fids.push_back(std::pair<size_t, double>(fid, fv));
                    if(opts.explain) {
                        nmap[fid] = fn;
                    }
                }
            }
            batchbuf.push_back(std::pair<double, std::vector<std::pair<size_t, double>>>(label, fids));
            descbuf.push_back(ssdesc.str());
            delete [] buff;
        }
        if(batchbuf.size() >= static_cast<size_t>(opts.batch) || (batchbuf.size() > 0 && line.empty())) {

            Batch batch;
            auto& nzf = batch.nzf;
            std::copy(descbuf.begin(), descbuf.end(), std::back_inserter(batch.desc));

            //nzf mapuje hash index featury na poradove cislo nenulove featury v batchi

            if(opts.explain) {
                batch.names.resize(nmap.size());
            }
            for(auto bit = batchbuf.begin(); bit != batchbuf.end(); ++bit) {
                auto& fids(bit->second);
                for(size_t i = 0; i < fids.size(); ++i) {
                    size_t id = nzf.size();
                    if(nzf.find(fids[i].first) == nzf.end()) {
                        //console->info("mapping = {0} -> {1}", fids[i].first, id);
                        nzf[fids[i].first] = id;
                        if(opts.explain) {
                            batch.names[id] = nmap[fids[i].first];
                        }
                    }
                }
            }

            const size_t K = nzf.size(), k = batchbuf.size();

            auto& x = batch.x;
            x.resize(k, K);
            x = MatrixXd::Zero(k, K);
            auto& labels = batch.labels;
            labels.resize(k, 1);

            //vyplneni dense matice nenulovych featur v batchi dle mapovani nzf
            for(auto bit = batchbuf.begin(); bit != batchbuf.end(); ++bit) {
                auto& fids(bit->second);
                double label(bit->first);
                size_t j = bit - batchbuf.begin();
                labels(j, 0) = label;
                for(size_t i = 0; i < fids.size(); ++i) {
                    x(j, nzf[fids[i].first]) = fids[i].second;
                    //console->info("{0} {1}", j, nzf[fids[i].first]);
                }
            }
            obuffer_.add(batch);
            batchbuf.clear();
            descbuf.clear();
        }
        if(line.empty())
            break;
    }
    if(interrupted)
        obuffer_.clear();
    obuffer_.add(S());
}

template class Stage2<std::string, Batch>;
