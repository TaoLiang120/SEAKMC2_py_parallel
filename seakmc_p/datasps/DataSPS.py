import copy
import time

import numpy as np
from mpi4py import MPI

import seakmc_p.datasps.PostSPS as postSPS
import seakmc_p.datasps.PreSPS as preSPS
import seakmc_p.datasps.SaddlePointSearch as mySPS
import seakmc_p.general.DataOut as Dataout
import seakmc_p.mpiconf.MPIconf as mympi
from seakmc_p.input.Input import MPI_Tags
from seakmc_p.restart.Restart import RESTART
from seakmc_p.spsearch.SaddlePoints import Data_SPs

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


def Write_RESTART(istep, finished_AVs, DefectBank_list, thisSuperBasin, seakmcdata, Eground,
                  thisDataSPs, AVitags, df_delete_SPs, undo_idavs, simulation_time):
    thisRestart = RESTART(istep, finished_AVs, DefectBank_list, thisSuperBasin, seakmcdata, Eground,
                          thisDataSPs, AVitags, df_delete_SPs, undo_idavs, simulation_time)
    thisRestart.to_file()
    thisRestart = None


def AV_is_done(idav, iav, idsps, ticav_l, thisAV_l, AVstring_l, thisSPS_l,
               Pre_Disps_l, isRecycled_l, isPGSYMM_l, thisSOPs_l, SNC_l, CalPref_l, thisVNS_l,
               thisDataSPs, AVitags, df_delete_SPs, undo_idavs, finished_AVs, simulation_time,
               istep, thissett, seakmcdata, DefectBank_list, thisSuperBasin, Eground,
               object_dict, Precursor=False, insituGSPS=False, WriteRestart=True):
    LogWriter = object_dict['LogWriter']
    DFWriter = object_dict['DFWriter']
    out_paths = object_dict['out_paths']
    ticav = ticav_l[iav]
    thisSPS_l[iav], df_delete_SPs = postSPS.SPs_1postprocessing(thissett, thisSPS_l[iav], df_delete_SPs, DFWriter,
                                                                nSPstart=thisSPS_l[iav].nSP, insituGSPS=insituGSPS)
    if Precursor:
        thisSPS_l[iav], df_delete_this = thisDataSPs.check_dup_avSP(idav, thisSPS_l[iav],
                                                                    seakmcdata.precursor_neighbors, AVitags,
                                                                    thissett.saddle_point)
    else:
        thisSPS_l[iav], df_delete_this = thisDataSPs.check_dup_avSP(idav, thisSPS_l[iav], seakmcdata.de_neighbors,
                                                                    AVitags, thissett.saddle_point)
    df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)

    if thisSPS_l[iav].nSP > 0:
        thisDataSPs = postSPS.insert_AVSP2DataSPs(thisDataSPs, thisSPS_l[iav], idav, DFWriter, Precursor=Precursor)
        if Precursor:
            if thissett.defect_bank["Recycle"]:
                DefectBank_list = postSPS.add_to_DefectBank(thissett, thisAV_l[iav], thisSPS_l[iav], isRecycled_l[iav],
                                                            isPGSYMM_l[iav], thisSOPs_l[iav].sch_symbol,
                                                            DefectBank_list, out_paths[3])
            AVstring_l[iav] += "\n" + f"Found {str(thisSPS_l[iav].nSP)} saddle points in {str(idav)} precursor volume!"
        else:
            Dataout.visualize_AV_SPs(thissett.visual, seakmcdata, AVitags, thisAV_l[iav], thisSPS_l[iav], istep, idav,
                                     out_paths[0])
            AVstring_l[iav] += "\n" + f"Found {str(thisSPS_l[iav].nSP)} saddle points in {str(idav)} active volume!"
        AVstring_l[iav] += "\n" + "-----------------------------------------------------------------"

    tocav = time.time()
    if Precursor:
        AVstring_l[
            iav] += "\n" + (f"Total time for {idav} precursor volume: "
                            f"{round(tocav - ticav_l[iav], thissett.system['float_precision'])} s")
    else:
        AVstring_l[
            iav] += "\n" + (f"Total time for {idav} active volume: "
                            f"{round(tocav - ticav_l[iav], thissett.system['float_precision'])} s")
    AVstring_l[iav] += "\n" + "-----------------------------------------------------------------"
    LogWriter.write_data(AVstring_l[iav])

    finished_AVs += 1
    undo_idavs = np.delete(undo_idavs, np.argwhere(undo_idavs == idav))
    if WriteRestart and thissett.system["Restart"]["WriteRestart"] and finished_AVs % thissett.system["Restart"][
        "AVStep4Restart"] == 0:
        Write_RESTART(istep, finished_AVs, DefectBank_list, thisSuperBasin, seakmcdata, Eground,
                      thisDataSPs, AVitags, df_delete_SPs, undo_idavs, simulation_time)

    ticav_l[iav] = None
    thisAV_l[iav] = None
    AVstring_l[iav] = None
    thisSPS_l[iav] = None
    Pre_Disps_l[iav] = None
    isRecycled_l[iav] = None
    isPGSYMM_l[iav] = None
    thisSOPs_l[iav] = None
    SNC_l[iav] = None
    CalPref_l[iav] = None
    thisVNS_l[iav] = None

    return thisDataSPs, AVitags, df_delete_SPs


#########################################################
def master_data_SPS(ntask_tot, nproc_task, ntask_time,
                    thisnspsearch, thisDataSPs, AVitags, df_delete_SPs, undo_idavs, finished_AVs, simulation_time,
                    this_idavs,
                    istep, thissett, seakmcdata, DefectBank_list, thisSuperBasin, Eground,
                    object_dict, Pre_Disps_l=None, PreLoadDisps=False, Precursor=False, insituGSPS=False,
                    WriteRestart=True):
    status = MPI.Status()
    LogWriter = object_dict['LogWriter']
    DFWriter = object_dict['DFWriter']
    float_precision = thissett.system['float_precision']
    navs = len(undo_idavs)
    if thissett.spsearch["TaskDist"].upper() == "SPS":
        processing_tasks = np.arange(ntask_tot, dtype=int)
    else:
        processing_tasks = mympi.get_2D_task_distribution(navs, thisnspsearch, ntask_time)

    isVisit = np.zeros(navs, dtype=bool)
    thisAV_l = [None] * navs
    AVstring_l = [""] * navs
    thisSPS_l = [None] * navs
    thisVNS_l = [None] * navs
    if not PreLoadDisps:
        Pre_Disps_l = [[] for _ in range(navs)]
        isRecycled_l = [False] * navs
    else:
        isRecycled_l = [True] * navs
    isPGSYMM_l = [False] * navs
    thisSOPs_l = [None] * navs
    SNC_l = [None] * navs
    CalPref_l = [None] * navs
    ticav_l = [None] * navs

    isDynmat_l = np.zeros(navs, dtype=bool)
    completed_avs = np.zeros(navs, dtype=bool)
    task_index = 0
    completed_tasks = 0
    completed_spsearchs_AV = np.zeros(navs, dtype=int)
    working_jobs = []
    while completed_tasks < ntask_tot:
        idtask = comm_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        s = status.Get_source()
        tag = status.Get_tag()
        thiscolor = int((s - 1) / nproc_task)
        ##rank_local = (s-1)%nproc_task
        if thiscolor in working_jobs:
            if tag == MPI_Tags.DONE:
                iav = int(idtask / thisnspsearch)
                idav = this_idavs[iav]
                idsps = idtask % thisnspsearch

                isVisit_this = comm_world.recv(source=s, tag=51)
                AVstring_l[iav] += comm_world.recv(source=s, tag=52)
                thisSP = comm_world.recv(source=s, tag=53)
                thisVN = comm_world.recv(source=s, tag=54)
                time_spsearch = comm_world.recv(source=s, tag=55)
                ntsiter = comm_world.recv(source=s, tag=56)
                if not isVisit_this:
                    isPGSYMM_l[iav] = comm_world.recv(source=s, tag=57)
                    thisSOPs_l[iav] = comm_world.recv(source=s, tag=58)
                    SNC_l[iav] = comm_world.recv(source=s, tag=59)
                    CalPref_l[iav] = comm_world.recv(source=s, tag=60)
                    isDynmat_l[iav] = comm_world.recv(source=s, tag=61)
                    if not PreLoadDisps:
                        Pre_Disps_l[iav] = comm_world.recv(source=s, tag=62)
                        isRecycled_l[iav] = comm_world.recv(source=s, tag=63)

                if thisVN is not None:
                    if len(thisVN) > 0 and len(thisVNS_l[iav]) <= thissett.spsearch["HandleVN"]["NMaxRandVN"]:
                        thisVNS_l[iav].append(thisVN)
                if thisSP is not None:
                    if thisSP.ISVALID:
                        isDup, df_delete_this = thisSPS_l[iav].check_this_duplicate(thisSP)
                        if isDup:
                            df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
                            thisSP.ISVALID = False
                        else:
                            if thissett.saddle_point["ValidSPs"]["RealtimeDelete"]:
                                df_delete_this, delete_this = thisSPS_l[iav].realtime_validate_thisSP(thisSP,
                                                                                                      insituGSPS=insituGSPS)
                                if delete_this:
                                    df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
                                    thisSP.ISVALID = False
                        if thisSP.ISVALID:
                            if thisSOPs_l[iav].nOP > 1:
                                thisSymmsps = thisSP.get_SPs_from_symmetry(
                                    thisAV_l[iav].to_coords(Buffer=False, Fixed=False), thisSOPs_l[iav])
                                thisSPS_l[iav].insert_SP([thisSP] + thisSymmsps)
                            else:
                                thisSPS_l[iav].insert_SP(thisSP)
                            if thissett.saddle_point["ValidSPs"]["RealtimeValid"]: thisSPS_l[iav].validate_SPs(
                                Delete=False, insituGSPS=insituGSPS)

                AVstring_l[
                    iav] += "\n" + (f"idAV:{idav}, idsps:{idsps},  ntrans: {ntsiter},  "
                                    f"barrier:{round(thisSP.barrier, float_precision)},  "
                                    f"ebias: {round(thisSP.ebias, float_precision)}")
                AVstring_l[
                    iav] += "\n" + (f"      dmag:{round(thisSP.dmag, float_precision)}, "
                                    f"dmagFin:{round(thisSP.fdmag, float_precision)}, isConnect: {thisSP.isconnect}")
                AVstring_l[
                    iav] += "\n" + (f"      Valid SPs: {thisSPS_l[iav].nvalid}, num of SPs:{thisSPS_l[iav].nSP}, "
                                    f"time: {round(time_spsearch, float_precision)} s")
                AVstring_l[iav] += "\n" + "-----------------------------------------------------------------"

                isVisit[iav] = True
                completed_spsearchs_AV[iav] += 1
                if completed_spsearchs_AV[iav] == thisnspsearch:
                    isAVDone = True
                else:
                    isAVDone = False

                if isAVDone:
                    thisDataSPs, AVitags, df_delete_SPs = AV_is_done(idav, iav, idsps, ticav_l, thisAV_l, AVstring_l,
                                                                     thisSPS_l,
                                                                     Pre_Disps_l, isRecycled_l, isPGSYMM_l, thisSOPs_l,
                                                                     SNC_l, CalPref_l, thisVNS_l,
                                                                     thisDataSPs, AVitags, df_delete_SPs, undo_idavs,
                                                                     finished_AVs, simulation_time,
                                                                     istep, thissett, seakmcdata, DefectBank_list,
                                                                     thisSuperBasin, Eground,
                                                                     object_dict, Precursor=Precursor,
                                                                     insituGSPS=insituGSPS, WriteRestart=WriteRestart)
                    completed_avs[iav] = True
                else:
                    pass

                working_jobs.remove(thiscolor)
                if task_index < ntask_tot: completed_tasks += 1

        else:
            if tag == MPI_Tags.READY:
                if task_index < ntask_tot:
                    idtask = processing_tasks[task_index]
                    iav = int(idtask / thisnspsearch)
                    idav = this_idavs[iav]
                    idsps = idtask % thisnspsearch
                    isVisit_this = isVisit[iav]

                    if task_index % (thissett.system['Interval4ShowProgress'] * ntask_time) == 0:
                        logstr = f"Processing the {idtask} task - idav: {idav} idsps: {idsps}."
                        LogWriter.write_data(logstr)

                    if ticav_l[iav] is None:
                        ticav_l[iav] = time.time()
                        thisAV = preSPS.initialize_thisAV(seakmcdata, idav, Rebuild=False, Precursor=Precursor)
                        local_coords, thisVNS_l[iav] = preSPS.initialize_AV_props(thisAV)
                        if thisAV is not None:
                            AVitags[idav] = thisAV.itags[0:(thisAV.nactive + thisAV.nbuffer)]
                            if Precursor:
                                x = seakmcdata.precursors.at[idav, 'xsn']
                                y = seakmcdata.precursors.at[idav, 'ysn']
                                z = seakmcdata.precursors.at[idav, 'zsn']
                                AVstring_l[
                                    iav] += "\n" + (f"Precursor ID: {idav} "
                                                    f"nactive:{thisAV.nactive} nbuffer:{thisAV.nbuffer} nfixed:{thisAV.nfixed}")
                                AVstring_l[
                                    iav] += "\n" + (f"Precursor center fractional coords: "
                                                    f"{round(x, 5), round(y, 5), round(z, 5)}")
                            else:
                                x = seakmcdata.defects.at[idav, 'xsn']
                                y = seakmcdata.defects.at[idav, 'ysn']
                                z = seakmcdata.defects.at[idav, 'zsn']
                                AVstring_l[
                                    iav] += "\n" + (f"ActVol ID: {idav} "
                                                    f"nactive:{thisAV.nactive} nbuffer:{thisAV.nbuffer} nfixed:{thisAV.nfixed}")
                                AVstring_l[
                                    iav] += "\n" + (f"AV center fractional coords: "
                                                    f"{round(x, 5), round(y, 5), round(z, 5)}")
                            thisSPS_l[iav] = preSPS.initialize_thisSPS(idav, local_coords, thisnspsearch, thissett)
                            SNC_l[iav], CalPref_l[iav], errorlog = preSPS.initial_SNC_CalPref(idav, thisAV, thissett,
                                                                                              Precursor=Precursor,
                                                                                              insituGSPS=insituGSPS)
                            if len(errorlog) > 0: AVstring_l[iav] += "\n" + errorlog
                        else:
                            AVstring_l[iav] += "\n" + f"Warning: ActiveVolume {idav} is NOT valid!"
                            AVstring_l[
                                iav] += "\n" + "-----------------------------------------------------------------"
                            thisSPS_l[iav] = preSPS.initialize_thisSPS(idav, np.array([[], [], []]), thisnspsearch,
                                                                       thissett)
                            SNC_l[iav] = False
                            CalPref_l[iav] = False

                        thisAV_l[iav] = thisAV

                    comm_world.send(idtask, dest=s, tag=MPI_Tags.START)
                    comm_world.send(isVisit_this, dest=s, tag=11)
                    comm_world.send(thisAV_l[iav], dest=s, tag=12)
                    comm_world.send(thisSPS_l[iav], dest=s, tag=13)
                    comm_world.send(thisVNS_l[iav], dest=s, tag=14)
                    comm_world.send(Pre_Disps_l[iav], dest=s, tag=15)
                    comm_world.send(isRecycled_l[iav], dest=s, tag=16)
                    comm_world.send(isPGSYMM_l[iav], dest=s, tag=17)
                    comm_world.send(thisSOPs_l[iav], dest=s, tag=18)
                    comm_world.send(SNC_l[iav], dest=s, tag=19)
                    comm_world.send(CalPref_l[iav], dest=s, tag=20)
                    comm_world.send(isDynmat_l[iav], dest=s, tag=21)

                    if task_index > 0 and task_index % ntask_time * 4 == 0:
                        ai = np.argwhere(completed_avs == True)
                        bi = np.argwhere(isDynmat_l == True)
                        to_delete_iavs = np.intersect1d(ai, bi)
                        comm_world.send(to_delete_iavs, dest=s, tag=22)
                    else:
                        comm_world.send(np.array([], dtype=int), dest=s, tag=22)
                    working_jobs.append(thiscolor)
                    task_index += 1
                else:
                    comm_world.send(None, dest=s, tag=MPI_Tags.EXIT)
            elif tag == MPI_Tags.EXIT:
                completed_tasks += 1

    return thisDataSPs, AVitags, df_delete_SPs


def slave_data_SPS(nproc_task, thiscolor, comm_split,
                   thisnspsearch, this_idavs, istep, thissett, DefectBank_list, dynmatAV_l,
                   object_dict, PreLoadDisps=False, Precursor=False, insituGSPS=False):
    status = MPI.Status()
    rank_local = comm_split.Get_rank()
    while True:
        if rank_local == 0:
            comm_world.send(None, dest=0, tag=MPI_Tags.READY)
            idtask = comm_world.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == MPI_Tags.START:
                isVisit_this = comm_world.recv(source=0, tag=11)
                thisAV = comm_world.recv(source=0, tag=12)
                thisSPS = comm_world.recv(source=0, tag=13)
                thisVNS = comm_world.recv(source=0, tag=14)
                Pre_Disps = comm_world.recv(source=0, tag=15)
                isRecycled = comm_world.recv(source=0, tag=16)
                isPGSYMM = comm_world.recv(source=0, tag=17)
                thisSOPs = comm_world.recv(source=0, tag=18)
                SNC = comm_world.recv(source=0, tag=19)
                CalPref = comm_world.recv(source=0, tag=20)
                isDynmat = comm_world.recv(source=0, tag=21)
                to_delete_iavs = comm_world.recv(source=0, tag=22)
            else:
                isVisit_this = None
                thisAV = None
                thisSPS = None
                thisVNS = None
                Pre_Disps = None
                isRecycled = None
                isPGSYMM = None
                thisSOPs = None
                SNC = None
                CalPref = None
                isDynmat = None
                to_delete_iavs = None
                #isRecv = None
                #dynmatAV = None
        else:
            idtask = None
            isVisit_this = None
            thisAV = None
            thisSPS = None
            thisVNS = None
            Pre_Disps = None
            isRecycled = None
            isPGSYMM = None
            thisSOPs = None
            SNC = None
            CalPref = None
            isDynmat = None
            to_delete_iavs = None
            tag = None
            #isRecv = None
            #dynmatAV = None

        if nproc_task > 1:
            tag = comm_split.bcast(tag, root=0)
            idtask = comm_split.bcast(idtask, root=0)
            isVisit_this = comm_split.bcast(isVisit_this, root=0)
            thisAV = comm_split.bcast(thisAV, root=0)
            thisSPS = comm_split.bcast(thisSPS, root=0)
            thisVNS = comm_split.bcast(thisVNS, root=0)
            Pre_Disps = comm_split.bcast(Pre_Disps, root=0)
            isRecycled = comm_split.bcast(isRecycled, root=0)
            isPGSYMM = comm_split.bcast(isPGSYMM, root=0)
            thisSOPs = comm_split.bcast(thisSOPs, root=0)
            SNC = comm_split.bcast(SNC, root=0)
            CalPref = comm_split.bcast(CalPref, root=0)
            isDynmat = comm_split.bcast(isDynmat, root=0)
            to_delete_iavs = comm_split.bcast(to_delete_iavs, root=0)
        if tag == MPI_Tags.START:
            for i in range(len(to_delete_iavs)):
                dynmatAV_l[to_delete_iavs[i]] = None

            iav = int(idtask / thisnspsearch)
            idav = this_idavs[iav]
            idsps = idtask % thisnspsearch
            AVstring = ""
            if thisAV is not None:
                if not isVisit_this:
                    if insituGSPS:
                        thisSOPs, isPGSYMM = preSPS.get_SymmOperators(thissett, thisAV, idav, PointGroup=False)
                    else:
                        thisSOPs, isPGSYMM = preSPS.get_SymmOperators(thissett, thisAV, idav,
                                                                      PointGroup=thissett.active_volume[
                                                                          "PointGroupSymm"])
                    if not PreLoadDisps:
                        isRecycled, Pre_Disps = preSPS.get_Pre_Disps(idav, thisAV, thissett, thisSOPs, DefectBank_list,
                                                                     istep)
                    #thisStrain = preSPS.get_AV_atom_strain(thisAV, thissett, thiscolor, comm=comm_split)

                    dynmatAV_l[iav] = None
                    if SNC or CalPref:
                        SNC, CalPref, dynmatAV_l[iav], isDynmat = preSPS.get_thisSNC4spsearch(idav, thissett, thisAV,
                                                                                              SNC, CalPref, object_dict,
                                                                                              thiscolor, istep,
                                                                                              comm=comm_split)
                        comm_split.Barrier()
                else:
                    if isDynmat and dynmatAV_l[iav] is None:
                        SNC, CalPref, dynmatAV_l[iav], isDynmat = preSPS.get_thisSNC4spsearch(idav, thissett, thisAV,
                                                                                              SNC, CalPref, object_dict,
                                                                                              thiscolor, istep,
                                                                                              comm=comm_split)
                        comm_split.Barrier()

                thisSP, thisVN, time_spsearch, ntsiter = mySPS.spsearch_search_single(nproc_task, thiscolor, comm_split,
                                                                                      istep, idav, thissett, thisAV, thisSOPs,
                                                                                      dynmatAV_l[iav], SNC, CalPref,
                                                                                      thisSPS,
                                                                                      Pre_Disps, idsps, thisVNS,
                                                                                      object_dict, Precursor=Precursor,
                                                                                      insituGSPS=insituGSPS)
            else:
                thisSP = None
                thisVN = []
                time_spsearch = 0.0
                ntsiter = 0
                if not isVisit_this:
                    thisSOPs = None
                    isPGSYMM = False
                    isRecycled = False
                    Pre_Disps = []
                    isDynmat = False
                    SNC = False
                    CalPref = False
                    #dynmatAV=None

            if rank_local == 0:
                comm_world.send(idtask, dest=0, tag=MPI_Tags.DONE)
                comm_world.send(isVisit_this, dest=0, tag=51)
                comm_world.send(AVstring, dest=0, tag=52)
                comm_world.send(thisSP, dest=0, tag=53)
                comm_world.send(thisVN, dest=0, tag=54)
                comm_world.send(time_spsearch, dest=0, tag=55)
                comm_world.send(ntsiter, dest=0, tag=56)
                if not isVisit_this:
                    comm_world.send(isPGSYMM, dest=0, tag=57)
                    comm_world.send(thisSOPs, dest=0, tag=58)
                    comm_world.send(SNC, dest=0, tag=59)
                    comm_world.send(CalPref, dest=0, tag=60)
                    comm_world.send(isDynmat, dest=0, tag=61)
                    if not PreLoadDisps:
                        comm_world.send(Pre_Disps, dest=0, tag=62)
                        comm_world.send(isRecycled, dest=0, tag=63)
            else:
                pass
        elif tag == MPI_Tags.EXIT:
            break

    if rank_local == 0: comm_world.send(None, dest=0, tag=MPI_Tags.EXIT)


def data_SPS_parrallel(nproc_task, start_proc, ntask_time, thiscolor, comm_split,
                       thisnspsearch, thisDataSPs, AVitags, df_delete_SPs, undo_idavs, finished_AVs, simulation_time,
                       istep, thissett, seakmcdata, DefectBank_list, thisSuperBasin, Eground,
                       object_dict, Pre_Disps_l=None, PreLoadDisps=False, Precursor=False, insituGSPS=False,
                       WriteRestart=True):
    ntask_tot = len(undo_idavs) * thisnspsearch
    ntask_time = min(ntask_time, ntask_tot)
    this_idavs = copy.deepcopy(undo_idavs)

    dynmatAV_l = [None] * len(undo_idavs)
    if rank_world == 0:
        thisDataSPs, AVitags, df_delete_SPs = master_data_SPS(ntask_tot, nproc_task, ntask_time,
                                                              thisnspsearch, thisDataSPs, AVitags, df_delete_SPs,
                                                              undo_idavs, finished_AVs, simulation_time, this_idavs,
                                                              istep, thissett, seakmcdata, DefectBank_list,
                                                              thisSuperBasin, Eground,
                                                              object_dict, Pre_Disps_l=Pre_Disps_l,
                                                              PreLoadDisps=PreLoadDisps, Precursor=Precursor,
                                                              insituGSPS=insituGSPS, WriteRestart=WriteRestart)
    else:
        if thiscolor < ntask_time:
            slave_data_SPS(nproc_task, thiscolor, comm_split,
                           thisnspsearch, this_idavs, istep, thissett, DefectBank_list, dynmatAV_l,
                           object_dict, PreLoadDisps=PreLoadDisps, Precursor=Precursor, insituGSPS=insituGSPS)
        else:
            pass

    thisAV = None
    thisSPS = None
    thisSOPs = None
    this_idavs = None
    dynmatAV = None
    dynmatAV_l = None

    comm_world.Barrier()

    return thisDataSPs, AVitags, df_delete_SPs


#################################################################################################
def Send_Results(thiscolor, iav, idav, idsps, thisSP, thisVN, time_spsearch, ntsiter, isSendOpt, thiscomm, PreLoadDisps,
                 *OptResults):
    itag = thiscolor * 100
    thiscomm.send(thiscolor, dest=0, tag=itag + 1)
    thiscomm.send(iav, dest=0, tag=itag + 2)
    thiscomm.send(idav, dest=0, tag=itag + 3)
    thiscomm.send(idsps, dest=0, tag=itag + 4)
    thiscomm.send(thisSP, dest=0, tag=itag + 5)
    thiscomm.send(thisVN, dest=0, tag=itag + 6)
    thiscomm.send(time_spsearch, dest=0, tag=itag + 7)
    thiscomm.send(ntsiter, dest=0, tag=itag + 8)
    thiscomm.send(isSendOpt, dest=0, tag=itag + 9)
    if isSendOpt:
        thiscomm.send(OptResults[0], dest=0, tag=itag + 10)
        thiscomm.send(OptResults[1], dest=0, tag=itag + 11)
        thiscomm.send(OptResults[2], dest=0, tag=itag + 12)
        thiscomm.send(OptResults[3], dest=0, tag=itag + 13)
        thiscomm.send(OptResults[4], dest=0, tag=itag + 14)
        thiscomm.send(OptResults[5], dest=0, tag=itag + 15)
        thiscomm.send(OptResults[6], dest=0, tag=itag + 16)
        thiscomm.send(OptResults[7], dest=0, tag=itag + 17)
        thiscomm.send(OptResults[8], dest=0, tag=itag + 18)
        thiscomm.send(OptResults[9], dest=0, tag=itag + 19)
        thiscomm.send(OptResults[10], dest=0, tag=itag + 20)
        if not PreLoadDisps:
            thiscomm.send(OptResults[11], dest=0, tag=itag + 21)
            thiscomm.send(OptResults[12], dest=0, tag=itag + 22)


###############################################
def data_SPS_no_master_slave(nproc_task, start_proc, ntask_time, thiscolor, comm_split,
                             thisnspsearch, thisDataSPs, AVitags, df_delete_SPs, undo_idavs, finished_AVs,
                             simulation_time,
                             istep, thissett, seakmcdata, DefectBank_list, thisSuperBasin, Eground,
                             object_dict, Pre_Disps_l=None, PreLoadDisps=False, Precursor=False, insituGSPS=False,
                             WriteRestart=True):
    rank_local = comm_split.Get_rank()
    float_precision = thissett.system['float_precision']
    LogWriter = object_dict['LogWriter']
    DFWriter = object_dict['DFWriter']

    this_idavs = copy.deepcopy(undo_idavs)
    navs = len(undo_idavs)
    ntask_tot = navs * thisnspsearch
    if thissett.spsearch["TaskDist"].upper() == "SPS":
        processing_tasks = np.arange(ntask_tot, dtype=int)
    else:
        processing_tasks = mympi.get_2D_task_distribution(navs, thisnspsearch, ntask_time)

    isVisit = np.zeros(navs, dtype=bool)
    thisAV_l = [None] * navs
    AVstring_l = [""] * navs
    AVstring_head_l = [""] * navs
    AVstring_app_l = [""] * navs
    local_coords_l = [None] * navs
    thisSPS_l = [None] * navs
    thisVNS_l = [None] * navs
    if not PreLoadDisps:
        Pre_Disps_l = [[] for _ in range(navs)]
        isRecycled_l = [False] * navs
    else:
        isRecycled_l = [True] * navs
    isPGSYMM_l = [False] * navs
    thisSOPs_l = [None] * navs
    SNC_l = np.zeros(navs, dtype=bool)
    CalPref_l = np.zeros(navs, dtype=bool)
    dynmatAV_l = [None] * navs
    ticav_l = [None] * navs
    completed_spsearchs_AV = np.zeros(navs, dtype=int)
    isDynmat_l = np.zeros(navs, dtype=bool)
    completed_avs = np.zeros(navs, dtype=bool)

    task_index = 0
    ntask_left = ntask_tot
    ntask_time = min(ntask_time, ntask_left)
    itask_start = 0
    iter = 0
    while ntask_left > 0:
        if thiscolor < ntask_time:
            idtask_local = itask_start + thiscolor
            idtask = processing_tasks[idtask_local]
            iav = int(idtask / thisnspsearch)
            idav = undo_idavs[iav]
            idsps = idtask % thisnspsearch

            idtask_0 = processing_tasks[itask_start]
            iav_0 = int(idtask_0 / thisnspsearch)
            iavs = [iav_0]
            isSend = np.zeros(ntask_time, dtype=bool)
            for i in range(ntask_time):
                ij = itask_start + i
                j = processing_tasks[ij]
                iavtmp = int(j / thisnspsearch)
                if iavtmp not in iavs:
                    iavs.append(iavtmp)
                    if isVisit[iavtmp]:
                        pass
                    else:
                        isSend[i] = True

            AVstring = ""
            if not isVisit[iav]:
                if rank_local == 0: ticav = time.time()
                thisAV = preSPS.initialize_thisAV(seakmcdata, idav, Rebuild=False, Precursor=Precursor)
                if thisAV is not None:
                    if rank_local == 0:
                        AVitags[idav] = thisAV.itags[0:(thisAV.nactive + thisAV.nbuffer)]
                        if Precursor:
                            x = seakmcdata.precursors.at[idav, 'xsn']
                            y = seakmcdata.precursors.at[idav, 'ysn']
                            z = seakmcdata.precursors.at[idav, 'zsn']
                            AVstring += (f"Precursor ID: {idav} "
                                         f"nactive:{thisAV.nactive} nbuffer:{thisAV.nbuffer} nfixed:{thisAV.nfixed}")
                            AVstring += "\n" + (f"Precursor center fractional coords: "
                                                f"{round(x, 5), round(y, 5), round(z, 5)}")
                        else:
                            x = seakmcdata.defects.at[idav, 'xsn']
                            y = seakmcdata.defects.at[idav, 'ysn']
                            z = seakmcdata.defects.at[idav, 'zsn']
                            AVstring += (f"ActVol ID: {idav} "
                                         f"nactive:{thisAV.nactive} nbuffer:{thisAV.nbuffer} nfixed:{thisAV.nfixed}")
                            AVstring += "\n" + (f"AV center fractional coords: "
                                                f"{round(x, 5), round(y, 5), round(z, 5)}")
                    local_coords, thisVNS = preSPS.initialize_AV_props(thisAV)
                    if insituGSPS:
                        thisSOPs, isPGSYMM = preSPS.get_SymmOperators(thissett, thisAV, idav, PointGroup=False)
                    else:
                        thisSOPs, isPGSYMM = preSPS.get_SymmOperators(thissett, thisAV, idav,
                                                                      PointGroup=thissett.active_volume[
                                                                          "PointGroupSymm"])

                    if not PreLoadDisps:
                        isRecycled, Pre_Disps = preSPS.get_Pre_Disps(idav, thisAV, thissett, thisSOPs, DefectBank_list,
                                                                     istep)
                    else:
                        Pre_Disps = Pre_Disps_l[iav]
                        isRecycled = isRecycled_l[iav]

                    thisSPS = preSPS.initialize_thisSPS(idav, local_coords, thisnspsearch, thissett)

                    SNC, CalPref, errorlog = preSPS.initial_SNC_CalPref(idav, thisAV, thissett, Precursor=Precursor,
                                                                        insituGSPS=insituGSPS)
                    if len(errorlog) > 0 and rank_local == 0: AVstring += "\n" + errorlog

                    #thisStrain = preSPS.get_AV_atom_strain(thisAV, thissett, thiscolor, comm=comm_split)
                    isDynmat = False
                    if SNC or CalPref:
                        SNC, CalPref, dynmatAV_l[iav], isDynmat = preSPS.get_thisSNC4spsearch(idav, thissett, thisAV,
                                                                                              SNC, CalPref, object_dict,
                                                                                              thiscolor, istep,
                                                                                              comm=comm_split)

                    comm_split.Barrier()

                    thisSP, thisVN, time_spsearch, ntsiter = mySPS.spsearch_search_single(nproc_task, thiscolor,
                                                                                          comm_split,
                                                                                          istep, idav, thissett, thisAV,
                                                                                          thisSOPs, dynmatAV_l[iav],
                                                                                          SNC, CalPref, thisSPS,
                                                                                          Pre_Disps, idsps, thisVNS,
                                                                                          object_dict,
                                                                                          Precursor=Precursor,
                                                                                          insituGSPS=insituGSPS)

                    if rank_local == 0:
                        OptResults = [thisAV, AVstring, ticav, thisSPS, local_coords,
                                      isPGSYMM, thisSOPs, SNC, CalPref, isDynmat, AVitags[idav], Pre_Disps, isRecycled]
                    else:
                        OptResults = []
                else:
                    if rank_local == 0:
                        AVstring += "\n" + f"Warning: ActiveVolume {idav} is NOT valid!"
                        AVstring += "\n" + "-----------------------------------------------------------------"
                    thisSPS = preSPS.initialize_thisSPS(idav, np.array([[], [], []]), thisnspsearch, thissett)
                    thisSP = None
                    thisVN = None
                    time_spsearch = 0
                    ntsiter = 0
                    local_coords = np.array([[], [], []])
                    if rank_local == 0:
                        OptResults = [None, AVstring, ticav, thisSPS, local_coords, None, None, False, False, False,
                                      np.array([], dtype=int), [], None]
                    else:
                        OptResults = []

                if thiscolor > 0 and rank_local == 0:
                    Send_Results(thiscolor, iav, idav, idsps, thisSP, thisVN, time_spsearch, ntsiter, isSend[thiscolor],
                                 comm_world, PreLoadDisps, *OptResults)
            else:
                if thisAV_l[iav] is not None:
                    if isDynmat_l[iav] and dynmatAV_l[iav] is None:
                        SNC_l[iav], CalPref_l[iav], dynmatAV_l[iav], isDynmat_l[iav] = (
                            preSPS.get_thisSNC4spsearch(idav, thissett,
                                                        thisAV_l[iav],
                                                        SNC_l[iav],
                                                        CalPref_l[iav],
                                                        object_dict,
                                                        thiscolor, istep,
                                                        comm=comm_split))
                        comm_split.Barrier()

                    thisSP, thisVN, time_spsearch, ntsiter = (
                        mySPS.spsearch_search_single(nproc_task, thiscolor,
                                                     comm_split,
                                                     istep, idav, thissett, thisAV_l[iav],
                                                     thisSOPs_l[iav],
                                                     dynmatAV_l[iav], SNC_l[iav],
                                                     CalPref_l[iav],
                                                     thisSPS_l[iav],
                                                     Pre_Disps_l[iav], idsps,
                                                     thisVNS_l[iav],
                                                     object_dict,
                                                     Precursor=Precursor,
                                                     insituGSPS=insituGSPS))

                    OptResults = []
                else:
                    thisSP = None
                    thisVN = None
                    time_spsearch = 0
                    ntsiter = 0
                    OptResults = []

                if thiscolor > 0 and rank_local == 0:
                    Send_Results(thiscolor, iav, idav, idsps, thisSP, thisVN, time_spsearch, ntsiter, False, comm_world,
                                 PreLoadDisps, *OptResults)
        else:
            thisSP = None
            thisVN = None
            time_spsearch = None
            ntsiter = None

        if rank_world == 0:
            if task_index % thissett.system['Interval4ShowProgress'] == 0:
                logstr = f"Processing the {idtask} task - idav: {idav} idsps: {idsps}."
                LogWriter.write_data(logstr)
            task_index += 1

            if not isVisit[iav]:
                thisAV_l[iav] = copy.deepcopy(thisAV)
                ticav_l[iav] = ticav
                thisSPS_l[iav] = copy.deepcopy(thisSPS)
                thisVNS_l[iav] = []
                local_coords_l[iav] = local_coords.copy()
                if thisAV_l[iav] is not None:
                    Pre_Disps_l[iav] = Pre_Disps.copy()
                    isRecycled_l[iav] = isRecycled
                    isPGSYMM_l[iav] = isPGSYMM
                    thisSOPs_l[iav] = copy.deepcopy(thisSOPs)
                    SNC_l[iav] = SNC
                    CalPref_l[iav] = CalPref
                    isDynmat_l[iav] = isDynmat
                    if len(AVstring_head_l[iav]) > 10:
                        pass
                    else:
                        AVstring_head_l[iav] += AVstring

            isVisit[iav] = True
            if thisVN is not None:
                if len(thisVN) > 0 and len(thisVNS_l[iav]) <= thissett.spsearch["HandleVN"]["NMaxRandVN"]:
                    thisVNS_l[iav].append(thisVN)

            if thisSP is not None:
                if thisSP.ISVALID:
                    isDup, df_delete_this = thisSPS_l[iav].check_this_duplicate(thisSP)
                    if isDup:
                        df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
                        thisSP.ISVALID = False
                    else:
                        if thissett.saddle_point["ValidSPs"]["RealtimeDelete"]:
                            df_delete_this, delete_this = thisSPS_l[iav].realtime_validate_thisSP(thisSP,
                                                                                                  insituGSPS=insituGSPS)
                            if delete_this:
                                df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
                                thisSP.ISVALID = False
                    if thisSP.ISVALID:
                        if thisSOPs_l[iav].nOP > 1:
                            thisSymmsps = thisSP.get_SPs_from_symmetry(local_coords_l[iav], thisSOPs_l[iav])
                            thisSPS_l[iav].insert_SP([thisSP] + thisSymmsps)
                        else:
                            thisSPS_l[iav].insert_SP(thisSP)
                        if thissett.saddle_point["ValidSPs"]["RealtimeValid"]:
                            thisSPS_l[iav].validate_SPs(Delete=False,
                                                        insituGSPS=insituGSPS)

                AVstring_app_l[
                    iav] += "\n" + (f"idAV:{idav}, idsps:{idsps},  ntrans: {ntsiter}, "
                                    f" barrier:{round(thisSP.barrier, float_precision)},  "
                                    f"ebias: {round(thisSP.ebias, float_precision)}")
                AVstring_app_l[
                    iav] += "\n" + (f"      dmag:{round(thisSP.dmag, float_precision)}, "
                                    f"dmagFin:{round(thisSP.fdmag, float_precision)}, isConnect: {thisSP.isconnect}")
                AVstring_app_l[
                    iav] += "\n" + (f"      Valid SPs: {thisSPS_l[iav].nvalid}, "
                                    f"num of SPs:{thisSPS_l[iav].nSP}, time: {round(time_spsearch, float_precision)} s")
                AVstring_app_l[iav] += "\n" + "-----------------------------------------------------------------"

            completed_spsearchs_AV[iav] += 1
            if completed_spsearchs_AV[iav] == thisnspsearch:
                AVstring_l[iav] = AVstring_head_l[iav] + AVstring_app_l[iav]
                thisDataSPs, AVitags, df_delete_SPs = AV_is_done(idav, iav, idsps, ticav_l, thisAV_l, AVstring_l,
                                                                 thisSPS_l,
                                                                 Pre_Disps_l, isRecycled_l, isPGSYMM_l, thisSOPs_l,
                                                                 SNC_l, CalPref_l, thisVNS_l,
                                                                 thisDataSPs, AVitags, df_delete_SPs, undo_idavs,
                                                                 finished_AVs, simulation_time,
                                                                 istep, thissett, seakmcdata, DefectBank_list,
                                                                 thisSuperBasin, Eground,
                                                                 object_dict, Precursor=Precursor,
                                                                 insituGSPS=insituGSPS, WriteRestart=WriteRestart)

                local_coords_l[iav] = None
                isDynmat_l[iav] = False
                AVstring_head_l[iav] = None
                AVstring_app_l[iav] = None
                completed_avs[iav] = True

            for i in range(1, ntask_time):
                itag = i * 100
                color_rec = comm_world.recv(source=i * nproc_task, tag=itag + 1)
                iav = comm_world.recv(source=i * nproc_task, tag=itag + 2)
                idav = comm_world.recv(source=i * nproc_task, tag=itag + 3)
                idsps = comm_world.recv(source=i * nproc_task, tag=itag + 4)
                thisSP = comm_world.recv(source=i * nproc_task, tag=itag + 5)
                thisVN = comm_world.recv(source=i * nproc_task, tag=itag + 6)
                time_spsearch = comm_world.recv(source=i * nproc_task, tag=itag + 7)
                ntsiter = comm_world.recv(source=i * nproc_task, tag=itag + 8)
                isSendOpt = comm_world.recv(source=i * nproc_task, tag=itag + 9)
                if isSendOpt:
                    thisAV_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 10)
                    AVstring = comm_world.recv(source=i * nproc_task, tag=itag + 11)
                    ticav_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 12)
                    thisSPS_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 13)
                    local_coords_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 14)
                    isPGSYMM_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 15)
                    thisSOPs_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 16)
                    SNC_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 17)
                    CalPref_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 18)
                    isDynmat_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 19)
                    thisAVitags = comm_world.recv(source=i * nproc_task, tag=itag + 20)
                    if not PreLoadDisps:
                        Pre_Disps_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 21)
                        isRecycled_l[iav] = comm_world.recv(source=i * nproc_task, tag=itag + 22)
                    thisVNS_l[iav] = []
                    if len(AVstring_head_l[iav]) > 10:
                        pass
                    else:
                        AVstring_head_l[iav] += AVstring
                    if thisAV_l[iav] is not None:
                        AVitags[idav] = copy.deepcopy(thisAVitags)

                isVisit[iav] = True
                if thisVN is not None:
                    if len(thisVN) > 0 and len(thisVNS_l[iav]) <= thissett.spsearch["HandleVN"]["NMaxRandVN"]:
                        thisVNS_l[iav].append(thisVN)
                if thisSP is not None:
                    if thisSP.ISVALID:
                        isDup, df_delete_this = thisSPS_l[iav].check_this_duplicate(thisSP)
                        if isDup:
                            df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
                            thisSP.ISVALID = False
                        else:
                            if thissett.saddle_point["ValidSPs"]["RealtimeDelete"]:
                                df_delete_this, delete_this = thisSPS_l[iav].realtime_validate_thisSP(thisSP,
                                                                                                      insituGSPS=insituGSPS)
                                if delete_this:
                                    df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
                                    thisSP.ISVALID = False
                        if thisSP.ISVALID:
                            if thisSOPs_l[iav].nOP > 1:
                                thisSymmsps = thisSP.get_SPs_from_symmetry(local_coords_l[iav], thisSOPs_l[iav])
                                thisSPS_l[iav].insert_SP([thisSP] + thisSymmsps)
                            else:
                                thisSPS_l[iav].insert_SP(thisSP)
                            if thissett.saddle_point["ValidSPs"]["RealtimeValid"]: thisSPS_l[iav].validate_SPs(
                                Delete=False, insituGSPS=insituGSPS)

                    AVstring_app_l[
                        iav] += "\n" + (f"idAV:{idav}, idsps:{idsps},  ntrans: {ntsiter},  "
                                        f"barrier:{round(thisSP.barrier, float_precision)},  "
                                        f"ebias: {round(thisSP.ebias, float_precision)}")
                    AVstring_app_l[
                        iav] += "\n" + (f"      dmag:{round(thisSP.dmag, float_precision)}, "
                                        f"dmagFin:{round(thisSP.fdmag, float_precision)}, isConnect: {thisSP.isconnect}")
                    AVstring_app_l[
                        iav] += "\n" + (f"      Valid SPs: {thisSPS_l[iav].nvalid}, num of SPs:{thisSPS_l[iav].nSP}, "
                                        f"time: {round(time_spsearch, float_precision)} s")
                    AVstring_app_l[iav] += "\n" + "-----------------------------------------------------------------"

                completed_spsearchs_AV[iav] += 1
                if completed_spsearchs_AV[iav] == thisnspsearch:
                    AVstring_l[iav] = AVstring_head_l[iav] + AVstring_app_l[iav]
                    thisDataSPs, AVitags, df_delete_SPs = AV_is_done(idav, iav, idsps, ticav_l, thisAV_l, AVstring_l,
                                                                     thisSPS_l,
                                                                     Pre_Disps_l, isRecycled_l, isPGSYMM_l, thisSOPs_l,
                                                                     SNC_l, CalPref_l, thisVNS_l,
                                                                     thisDataSPs, AVitags, df_delete_SPs, undo_idavs,
                                                                     finished_AVs, simulation_time,
                                                                     istep, thissett, seakmcdata, DefectBank_list,
                                                                     thisSuperBasin, Eground,
                                                                     object_dict, Precursor=Precursor,
                                                                     insituGSPS=insituGSPS, WriteRestart=WriteRestart)
                    local_coords_l[iav] = None
                    isDynmat_l[iav] = False
                    AVstring_head_l[iav] = None
                    AVstring_app_l[iav] = None
                    completed_avs[iav] = True
        else:
            pass

        comm_world.Barrier()

        isVisit = comm_world.bcast(isVisit, root=0)
        thisAV_l = comm_world.bcast(thisAV_l, root=0)
        thisSPS_l = comm_world.bcast(thisSPS_l, root=0)
        local_coords_l = comm_world.bcast(local_coords_l, root=0)
        Pre_Disps_l = comm_world.bcast(Pre_Disps_l, root=0)
        isRecycled_l = comm_world.bcast(isRecycled_l, root=0)
        isPGSYMM_l = comm_world.bcast(isPGSYMM_l, root=0)
        thisSOPs_l = comm_world.bcast(thisSOPs_l, root=0)
        SNC_l = comm_world.bcast(SNC_l, root=0)
        CalPref_l = comm_world.bcast(CalPref_l, root=0)
        isDynmat_l = comm_world.bcast(isDynmat_l, root=0)
        thisVNS_l = comm_world.bcast(thisVNS_l, root=0)

        if iter > 0 and iter % 4 == 0:
            completed_avs = comm_world.bcast(completed_avs, root=0)
            ai = np.argwhere(completed_avs == True)
            bi = np.argwhere(isDynmat_l == True)
            ci = np.intersect1d(ai, bi)
            for i in ci:
                dynmatAV_l[i] = None

        itask_start += ntask_time
        ntask_left = ntask_left - ntask_time
        ntask_time = min(ntask_time, ntask_left)
        iter += 1

    thisAV = None
    thisSPS = None
    thisSOPs = None
    this_idavs = None
    dynmatAV = None
    dynmatAV_l = None

    comm_world.Barrier()

    return thisDataSPs, AVitags, df_delete_SPs


#####################################################
def data_find_saddlepoints(istep, thissett, seakmcdata, DefectBank_list, thisSuperBasin, Eground,
                           DataSPs, AVitags, df_delete_SPs, undo_idavs, finished_AVs, simulation_time, object_dict):
    nproc_task = thissett.spsearch["force_evaluator"]["nproc"]
    Master_Slave = thissett.spsearch["Master_Slave"]
    if Master_Slave:
        start_proc = 1
    else:
        start_proc = 0
    if nproc_task == size_world:
        start_proc = 0
        Master_Slave = False

    ntask_time = mympi.get_ntask_time(nproc_task, start_proc=start_proc, thiscomm=None)
    comm_split, thiscolor = mympi.split_communicator(nproc_task, start_proc=start_proc, thiscomm=None)

    preSPS.initialization_thisdata(seakmcdata, thissett)
    LogWriter = object_dict['LogWriter']
    if thissett.spsearch["insituGuidedSPS"] and seakmcdata.nprecursors > 0:
        if rank_world == 0:
            logstr = f"There are total {seakmcdata.nprecursors} precursor volumes."
            logstr += "\n" + "-----------------------------------------------------------------"
            LogWriter.write_data(logstr)

        for idav in undo_idavs:
            emptya = np.array([], dtype=int)
            thisAVitags = [emptya for i in range(seakmcdata.nprecursors)]
            thisDataSPs = Data_SPs(istep, seakmcdata.nprecursors)
            thisDataSPs.initialization()
            thisundo_idavs = seakmcdata.def_atoms[idav]["itag"]
            thisfinished_AVs = 0
            thisnspsearch = thissett.spsearch["NSearch"]

            if rank_world == 0:
                logstr = f"The precursor IDs: {thisundo_idavs} AV ID:{idav}."
                logstr += "\n" + "-----------------------------------------------------------------"
                LogWriter.write_data(logstr)

            if Master_Slave:
                thisDataSPs, thisAVitags, df_delete_SPs = data_SPS_parrallel(nproc_task, start_proc, ntask_time,
                                                                             thiscolor, comm_split,
                                                                             thisnspsearch, thisDataSPs, thisAVitags,
                                                                             df_delete_SPs, thisundo_idavs,
                                                                             thisfinished_AVs, simulation_time,
                                                                             istep, thissett, seakmcdata,
                                                                             DefectBank_list, thisSuperBasin, Eground,
                                                                             object_dict, Pre_Disps_l=None,
                                                                             PreLoadDisps=False, Precursor=True,
                                                                             insituGSPS=False, WriteRestart=False)
            else:
                thisDataSPs, thisAVitags, df_delete_SPs = data_SPS_no_master_slave(nproc_task, start_proc, ntask_time,
                                                                                   thiscolor, comm_split,
                                                                                   thisnspsearch, thisDataSPs,
                                                                                   thisAVitags, df_delete_SPs,
                                                                                   thisundo_idavs, thisfinished_AVs,
                                                                                   simulation_time,
                                                                                   istep, thissett, seakmcdata,
                                                                                   DefectBank_list, thisSuperBasin,
                                                                                   Eground,
                                                                                   object_dict, Pre_Disps_l=None,
                                                                                   PreLoadDisps=False, Precursor=True,
                                                                                   insituGSPS=False, WriteRestart=False)

            comm_world.Barrier()

            if rank_world == 0:
                thisAV = preSPS.initialize_thisAV(seakmcdata, idav, Rebuild=False, Precursor=False)
                tmpAVitags = thisAV.itags[0:(thisAV.nactive + thisAV.nbuffer)]
                Pre_Disps_l = []
                #print(f"thisdata.nsp: {thisDataSPs.nSP}")
                isRecycled, Pre_Disps, precursorbarrs = preSPS.get_Pre_Disps_from_precursor(idav, thisAV, thissett,
                                                                                            seakmcdata, tmpAVitags,
                                                                                            thisDataSPs, thisAVitags)
                thisnsps = len(Pre_Disps)
                if isinstance(thissett.spsearch["NMaxSPs4insituGSPS"], int):
                    thisNMaxSPs = max(int(thissett.spsearch["NMaxSPs4insituGSPS"] / len(undo_idavs)), 1)
                    if len(Pre_Disps) > thisNMaxSPs:
                        ai = np.argsort(precursorbarrs)
                        ai = ai[0:thisNMaxSPs]
                        Pre_Disps = list(map(lambda i: Pre_Disps[i], ai))
                        ai = []
                else:
                    pass
                precursorbarrs = []
                thisnspsearch = len(Pre_Disps)
                Pre_Disps_l.append(Pre_Disps)

                logstr = "\n" + f"Finished SPS on precursor volumes of {idav} AV and found {thisnsps} saddle points."
                logstr += "\n" + f"The first {thisnspsearch} lowest barrier saddle points are subjected to insituGSPS."
                logstr += "\n" + "-----------------------------------------------------------------"
                LogWriter.write_data(logstr)
            else:
                Pre_Disps_l = None
                thisnspsearch = None

            comm_world.Barrier()
            Pre_Disps_l = comm_world.bcast(Pre_Disps_l, root=0)
            thisnspsearch = comm_world.bcast(thisnspsearch, root=0)
            thisAVitags = []
            thisDataSPs = None

            thisundo_idavs = np.array([idav])
            thisfinished_AVs = 0
            if Master_Slave:
                DataSPs, AVitags, df_delete_SPs = data_SPS_parrallel(nproc_task, start_proc, ntask_time, thiscolor,
                                                                     comm_split,
                                                                     thisnspsearch, DataSPs, AVitags, df_delete_SPs,
                                                                     thisundo_idavs, thisfinished_AVs, simulation_time,
                                                                     istep, thissett, seakmcdata, DefectBank_list,
                                                                     thisSuperBasin, Eground,
                                                                     object_dict, Pre_Disps_l=Pre_Disps_l,
                                                                     PreLoadDisps=True, Precursor=False,
                                                                     insituGSPS=True, WriteRestart=False)
            else:
                DataSPs, AVitags, df_delete_SPs = data_SPS_no_master_slave(nproc_task, start_proc, ntask_time,
                                                                           thiscolor, comm_split,
                                                                           thisnspsearch, DataSPs, AVitags,
                                                                           df_delete_SPs, thisundo_idavs,
                                                                           thisfinished_AVs, simulation_time,
                                                                           istep, thissett, seakmcdata, DefectBank_list,
                                                                           thisSuperBasin, Eground,
                                                                           object_dict, Pre_Disps_l=Pre_Disps_l,
                                                                           PreLoadDisps=True, Precursor=False,
                                                                           insituGSPS=True, WriteRestart=False)

            Pre_Disps_l = []
            undo_idavs = np.delete(undo_idavs, np.where(undo_idavs == idav))
            finished_AVs += 1

            if rank_world == 0:
                if (thissett.system["Restart"]["WriteRestart"] and
                        finished_AVs % thissett.system["Restart"][
                            "AVStep4Restart"] == 0):
                    Write_RESTART(istep, finished_AVs, DefectBank_list, thisSuperBasin, seakmcdata, Eground,
                                  DataSPs, AVitags, df_delete_SPs, undo_idavs, simulation_time)
            comm_world.Barrier()
    else:
        thisnspsearch = thissett.spsearch["NSearch"]
        if Master_Slave:
            DataSPs, AVitags, df_delete_SPs = data_SPS_parrallel(nproc_task, start_proc, ntask_time, thiscolor,
                                                                 comm_split,
                                                                 thisnspsearch, DataSPs, AVitags, df_delete_SPs,
                                                                 undo_idavs, finished_AVs, simulation_time,
                                                                 istep, thissett, seakmcdata, DefectBank_list,
                                                                 thisSuperBasin, Eground,
                                                                 object_dict, Pre_Disps_l=None, PreLoadDisps=False,
                                                                 Precursor=False, insituGSPS=False, WriteRestart=True)
        else:
            DataSPs, AVitags, df_delete_SPs = data_SPS_no_master_slave(nproc_task, start_proc, ntask_time, thiscolor,
                                                                       comm_split,
                                                                       thisnspsearch, DataSPs, AVitags, df_delete_SPs,
                                                                       undo_idavs, finished_AVs, simulation_time,
                                                                       istep, thissett, seakmcdata, DefectBank_list,
                                                                       thisSuperBasin, Eground,
                                                                       object_dict, Pre_Disps_l=None,
                                                                       PreLoadDisps=False, Precursor=False,
                                                                       insituGSPS=False, WriteRestart=True)

    comm_split.Free()
    comm_world.Barrier()

    if rank_world == 0:
        pass
    else:
        DataSPs = None
        AVitags = None
        df_delete_SPs = None

    DataSPs = comm_world.bcast(DataSPs, root=0)
    AVitags = comm_world.bcast(AVitags, root=0)
    df_delete_SPs = comm_world.bcast(df_delete_SPs, root=0)
    comm_world.Barrier()

    return DataSPs, AVitags, df_delete_SPs
