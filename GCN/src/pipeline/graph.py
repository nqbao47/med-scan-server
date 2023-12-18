import itertools
import math
import os
import re

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Grapher:
    """
    Description:
    ===========
            This class is used to generate:
                    1) the graph (in dictionary form) { source_node: [destination_node1, destination_node2]}
                    2) the dataframe with relative_distances

    Inputs: The class consists of a pandas dataframe consisting of cordinates for bounding boxe and the image of the invoice/receipt.

    """

    def __init__(self, filename):
        self.filename = filename
        file_path = (
            "D:/Luan Van/Project/med-scan-backend/processing/OCR_model/"
            + filename
            + ".csv"
        )
        image_path = "D:/Luan Van/Project/med-scan-backend/input/" + filename + ".jpg"
        self.df = pd.read_csv(file_path, header=None, sep="\t")
        self.image = cv2.imread(image_path)
        # Initialize TfidfVectorizer
        self.vectorizer = TfidfVectorizer()

    def graph_formation(self, export_graph=False):
        """
        Description:
        ===========
        Line formation:
        1) Sort words based on Top coordinate:
        2) Form lines as group of words which obeys the following:
            Two words (W_a and W_b) are in same line if:
                Top(W_a) <= Bottom(W_b) and Bottom(W_a) >= Top(W_b)
        3) Sort words in each line based on Left coordinate

        This ensures that words are read from top left corner of the image first,
        going line by line from left to right and at last the final bottom right word of the page is read.

        Args:
            df with words and cordinates (xmin,xmax,ymin,ymax)
            image read into cv2
        returns:
            df with words arranged in orientation top to bottom and left to right, the line number for each word, index of the node connected to
            on all directions top, bottom, right and left (if they exist and satisfy the parameters provided)

        _____________________y axis______________________
        |
        |                       top
        x axis               ___________________
        |              left | bounding box      |  right
        |                   |___________________|
        |                       bottom
        |
        |


        iterate through the rows twice to compare them.
        remember that the axes are inverted.

        """
        df, image = self.df, self.image
        """
        preprocessing the raw csv files to favorable df
        """
        # df = pd.read_csv(self.filepath, header=None, sep="\t")
        df = df[0].str.split(",", expand=True)
        temp = df.copy()
        temp[temp.columns] = temp.apply(lambda x: x.str.strip())
        temp.fillna("", inplace=True)
        temp[8] = temp[8].str.cat(temp.iloc[:, 9:], sep=", ")
        temp[temp.columns] = temp.apply(lambda x: x.str.rstrip(", ,"))
        temp = temp.loc[:, :8]
        temp.drop([2, 3, 6, 7], axis=1, inplace=True)
        temp.columns = ["xmin", "ymin", "xmax", "ymax", "Object"]

        # Thêm cột "labels" từ df_withlabels
        # temp["labels"] = self.df_withlabels["9"].astype(str)

        temp[["xmin", "ymin", "xmax", "ymax"]] = temp[
            ["xmin", "ymin", "xmax", "ymax"]
        ].apply(pd.to_numeric)

        df = temp

        # print("df first")
        # print(df)
        assert (
            type(df) == pd.DataFrame
        ), f"object_map should be of type \
            {pd.DataFrame}. Received {type(df)}"
        assert (
            type(image) == np.ndarray
        ), f"image should be of type {np.ndarray} \
            . Received {type(image)}"

        assert "xmin" in df.columns, '"xmin" not in object map'
        assert "xmax" in df.columns, '"xmax" not in object map'
        assert "ymin" in df.columns, '"ymin" not in object map'
        assert "ymax" in df.columns, '"ymax" not in object map'
        assert "Object" in df.columns, '"Object" column not in object map'

        # remove empty spaces both in front and behind
        for col in df.columns:
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass
        # for col in df.columns:
        #     if df[col].dtype == "object":
        #         df[col] = df[col].str.strip()

        # further cleaning
        df.dropna(inplace=True)
        # sort from top to bottom
        df.sort_values(by=["ymin"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # subtracting ymax by 1 to eliminate ambiguity of boxes being in both left and right
        df["ymax"] = df["ymax"].apply(lambda x: x - 1)

        master = []
        for idx, row in df.iterrows():
            # flatten the nested list
            flat_master = list(itertools.chain(*master))
            # check to see if idx is in flat_master
            if idx not in flat_master:
                top_a = row["ymin"]
                bottom_a = row["ymax"]
                # every line will atleast have the word in it
                line = [idx]
                for idx_2, row_2 in df.iterrows():
                    # check to see if idx_2 is in flat_master removes ambiguity
                    # picks higher cordinate one.
                    if idx_2 not in flat_master:
                        # if not the same words
                        if not idx == idx_2:
                            top_b = row_2["ymin"]
                            bottom_b = row_2["ymax"]
                            if (top_a <= bottom_b) and (bottom_a >= top_b):
                                line.append(idx_2)
                master.append(line)
        df2 = pd.DataFrame(
            {
                "words_indices": master,
                "line_number": [x for x in range(1, len(master) + 1)],
            }
        )

        # explode the list columns eg : [1,2,3]
        df2 = (
            df2.set_index("line_number")
            .words_indices.apply(pd.Series)
            .stack()
            .reset_index(level=0)
            .rename(columns={0: "words_indices"})
        )
        df2["words_indices"] = df2["words_indices"].astype("int")
        # put the line numbers back to the list
        final = df.merge(df2, left_index=True, right_on="words_indices")
        final.drop("words_indices", axis=1, inplace=True)

        """
        3) Sort words in each line based on Left coordinate
        """
        final2 = (
            final.sort_values(by=["line_number", "xmin"], ascending=True)
            .groupby("line_number")
            .head(len(final))
            .reset_index(drop=True)
        )

        df = final2
        # print("bao", df)
        """
        Pseudocode:
        1) Read words from each line starting from topmost line going towards bottommost line
        2) For each word, perform the following:
            - Check words which are in vertical projection with it.
            - Calculate RD_l and RD_r for each of them
            - Select nearest neighbour words in horizontal direction which have least magnitude of RD_l and RD_r,
            provided that those words do not have an edge in that direciton.
                    - In case, two words have same RD_l or RD_r, the word having higher top coordinate is chosen.
            - Repeat steps from 2.1 to 2.3 similarly for retrieving nearest neighbour words in vertical direction by
            taking horizontal projection, calculating RD_t and RD_b and choosing words having higher left co-ordinate
            incase of ambiguity
            - Draw edges between word and its 4 nearest neighbours if they are available.

        Args:
            df after lines properly aligned

        returns:
            graph in the form of a dictionary, networkX graph, dataframe with

        """

        # horizontal edges formation
        # print(df)
        df.reset_index(inplace=True)
        grouped = df.groupby("line_number")
        # for undirected graph construction
        horizontal_connections = {}
        # left
        left_connections = {}
        # right
        right_connections = {}

        for _, group in grouped:
            a = group["index"].tolist()
            b = group["index"].tolist()
            horizontal_connection = {a[i]: a[i + 1] for i in range(len(a) - 1)}
            # storing directional connections
            right_dict_temp = {a[i]: {"right": a[i + 1]} for i in range(len(a) - 1)}
            left_dict_temp = {b[i + 1]: {"left": b[i]} for i in range(len(b) - 1)}

            # add the indices in the dataframes
            for i in range(len(a) - 1):
                df.loc[df["index"] == a[i], "right"] = int(a[i + 1])
                df.loc[df["index"] == a[i + 1], "left"] = int(a[i])

            left_connections.update(right_dict_temp)
            right_connections.update(left_dict_temp)
            horizontal_connections.update(horizontal_connection)

        dic1, dic2 = left_connections, right_connections

        # Initialize "right" and "left" columns in the DataFrame
        # df["right"] = np.NaN
        # df["left"] = np.NaN

        # verticle connections formation
        bottom_connections = {}
        top_connections = {}

        for idx, row in df.iterrows():
            if idx not in bottom_connections.keys():
                right_a = row["xmax"]
                left_a = row["xmin"]

                for idx_2, row_2 in df.iterrows():
                    # check for higher idx values

                    if idx_2 not in bottom_connections.values() and idx < idx_2:
                        right_b = row_2["xmax"]
                        left_b = row_2["xmin"]
                        if (left_b <= right_a) and (right_b >= left_a):
                            bottom_connections[idx] = idx_2
                            top_connections[idx_2] = idx

                            # # Update the "right" and "left" columns
                            # df.loc[df["index"] == idx, "right"] = idx_2
                            # df.loc[df["index"] == idx_2, "left"] = idx
                            # add it to the dataframe
                            df.loc[df["index"] == idx, "bottom"] = idx_2
                            df.loc[df["index"] == idx_2, "top"] = idx
                            # print(bottom_connections)
                            # once the condition is met, break the loop to reduce redundant time complexity
                            break

        # combining both
        result = {}
        dic1 = horizontal_connections
        dic2 = bottom_connections

        for key in dic1.keys() | dic2.keys():
            if key in dic1:
                result.setdefault(key, []).append(dic1[key])
            if key in dic2:
                result.setdefault(key, []).append(dic2[key])
        # print(result)

        G = nx.from_dict_of_lists(result)

        if export_graph:
            if not os.path.exists("../../figures/graphs"):
                os.makedirs("../../figures/graphs")

            plot_path = "../../figures/graphs/" + self.filename + "plain_graph" ".jpg"
            print(plot_path)

            # np.random.seed(42)
            layout = nx.kamada_kawai_layout(G)
            # layout = nx.spring_layout(G, seed=42)

            nx.draw(G, layout, with_labels=True)
            plt.savefig(plot_path, format="jpg", dpi=600)
            plt.show()

        # print("self.df_withlabels ORI")
        # print(self.df_withlabels)

        # # connect with the labeling file that has labels in it
        # df["labels"] = self.df_withlabels["9"]
        # df["labels"] = df["labels"].replace("nan", np.nan)
        self.df = df
        # print(self.df.dtypes)

        # print(df)
        return G, result, df

    # features calculation
    def get_text_features(self, df):
        """
        gets text features

        Args: df
        Returns: n_lower, n_upper, n_spaces, n_alpha, n_numeric,n_special
        """
        data = df["Object"].tolist()
        # Danh sách các tên thuốc
        drug_names = [
            "Nebivolo",
            "Diastase",
            "Fexofena",
            "methylprednisolon",
            "Spironolactone",
            "Rabeprazol",
            "Rosuvastatin",
            "Propylthiouracil",
            "Metfomin",
            "Levosupind",
            "Levosulpirid",
            "Rabeprazo",
            "Rabeprazol",
            "URSODIO",
            "Ursodiol",
            "Rabepfazol",
            "Clopidogrel",
            "Clopidogre1",
            "Acetylsalicylic",
            "Acetylcystein",
            "Acetylcystein",
            "Acetylcystein",
            "AcetylCystein",
            "Agilodin",
            "Agiparofen",
            "Aleucin",
            "Allvitamine",
            "Altamin",
            "Amcinol",
            "Ampelop",
            "Antilox",
            "Avarino",
            "Avircrem",
            "Becacold",
            "Becacold",
            "Benita",
            "Berberin",
            "Berocca",
            "Betadine",
            "Biafine",
            "Bidivon",
            "Bifehema",
            "Bisacodyl",
            "Bisbeta",
            "BOCINOR",
            "Boganic",
            "Bosfen",
            "Bostanex",
            "Tydol",
            "Brometic",
            "Bromhexin",
            "Bundle",
            "Boston",
            "Calamine",
            "Calcium",
            "Calci",
            "Canesten",
            "Canesten",
            "Salonsip",
            "Salonpas",
            "Salonpas",
            "Tiger",
            "Salonpas",
            "Tiger",
            "Carflem",
            "Cerciorat",
            "Cetimed",
            "Cetirizine",
            "Clorpheniramin",
            "Coldacmin",
            "Colocol",
            "Alcool",
            "Boric",
            "Contractubex",
            "Acool",
            "Creon",
            "Daflavon",
            "Daflon",
            "Brand",
            "Eagle",
            "Brand",
            "Nizoral",
            "Nizoral",
            "Nizoral",
            "Selsun",
            "Mekophar",
            "Eagle",
            "Davita",
            "Desbebe",
            "Descallerg",
            "Desloratadin",
            "Dicenin",
            "Dolnaltic",
            "Domela",
            "Domperidon",
            "Dompidone",
            "Rectiofar",
            "Natri",
            "Systane",
            "Denicol",
            "Betadine",
            "LeoPovidone",
            "Povidine",
            "Lactulose",
            "Zyrtec",
            "Phytogyno",
            "Phytogyno",
            "Eagle",
            "Ebysta",
            "Effer",
            "Effer",
            "Enterogran",
            "Essentiale",
            "Eugica",
            "Eyelight",
            "Eyelight",
            "Eyetamin",
            "Fefasdin",
            "Fegra",
            "Fegra",
            "Ferlatum",
            "Fexofenadin",
            "Fexostad",
            "Fortrans",
            "Fugacar",
            "Gamalate",
            "Remos",
            "Salonpas",
            "LeoPovidone",
            "Mydugyno",
            "Giloba",
            "Gimfastnew",
            "Glotadol",
            "Glotadol",
            "Glucosamin",
            "Gynapax",
            "Halixol",
            "Halixol",
            "Hapacol",
            "Hapacol",
            "Hapacol",
            "Hapacol",
            "Hesmin",
            "Imexophen",
            "Jazxylo",
            "Kacerin",
            "Kanausin",
            "Katrypsin",
            "Calcrem",
            "Panthenol",
            "Silvirin",
            "Sulfadiazin",
            "Kamistad",
            "Voltaren",
            "Sulfadiazin",
            "Kamistad",
            "Voltaren",
            "Terbinafine",
            "Tyrosur",
            "Canesten",
            "Legalon",
            "Lessenol",
            "Lessenol",
            "Levoagi",
            "Livolin",
            "Livolin",
            "Loperamid",
            "MAGNE",
            "Magnesi",
            "Mebendazol",
            "Mezapulgit",
            "Mimosa",
            "MOGASTIC",
            "Molitoux",
            "Mucome",
            "Naphacogyl",
            "Nebivoloi",
            "Nasol",
            "Natri",
            "Natri",
            "Natri",
            "Natri",
            "Natri",
            "Neotica",
            "Newstomaz",
            "Nidal",
            "Normagut",
            "Nostravin",
            "Obimin",
            "Ocehepa",
            "Oralegic",
            "Oresol",
            "Panadol",
            "Paracetamol",
            "PARACETAMOL",
            "Paralmax",
            "Philcotam",
            "Phospha",
            "VITAMIN",
            "Povidone",
            "Povidon",
            "Progermila",
            "Prospan",
            "Remint",
            "Remos",
            "Remowart",
            "Reprat",
            "Rosiden",
            "Rotundin",
            "Rotundin",
            "Salonpas",
            "Salonpas",
            "Semiflit",
            "Siang",
            "Siang",
            "Spironolacton",
            "Bostanex",
            "Brufen",
            "Danospan",
            "Ambroco",
            "Pectol",
            "HoAstex",
            "HoAstex",
            "Smecta",
            "SnowClear",
            "Stacytine",
            "Stoccel",
            "Strepsils",
            "Sucrapi",
            "Tatanol",
            "Tatanol",
            "Tazoretin",
            "Tebonin",
            "Telfast",
            "Tezkin",
            "Dogarlic",
            "Stilux",
            "Tonka",
            "Boganic",
            "Logpatat",
            "Cebraton",
            "Calci",
            "Caldihasan",
            "Calcium",
            "Calcium",
            "Briozcal",
            "Farzincol",
            "Enpovid",
            "Obimin",
            "Bioflora",
            "Oresol",
            "Hydrite",
            "Hydrite",
            "Buscopan",
            "Dimenhydrinat",
            "Cinnarizin",
            "Acemuc",
            "Acemuc",
            "CalciD",
            "Gaviscon",
            "Liverton",
            "Liverton",
            "Ursimex",
            "Hexinvon",
            "Calcium",
            "Ferrovit",
            "Vitamin",
            "Orlistat",
            "Vitamin",
            "Acetylcystein",
            "Biotin",
            "Hapacol",
            "Mecaflu",
            "Eugica",
            "Eugica",
            "Gaviscon",
            "Benda",
            "Acemuc",
            "Tiffy",
            "Efferalgan",
            "Efferalgan",
            "Efferalgan",
            "Stadeltine",
            "Aerius",
            "Aerius",
            "Telfor",
            "Tottri",
            "Paralmax",
            "Andol",
            "Duspatalin",
            "Espumisan",
            "Topralsin",
            "Tanganil",
            "Phosphalugel",
            "Kremil",
            "Mefenamic",
            "Tydol",
            "Neopeptine",
            "Espumisan",
            "GazGo",
            "Allermine",
            "Erolin",
            "Tardyferon",
            "Bidiferon",
            "SnowClear",
            "Alaxan",
            "Panadol",
            "Alzental",
            "MediEucalyptol",
            "Cedipect",
            "Terpinzoat",
            "Lorastad",
            "Bromhexin",
            "Bromhexin",
            "Bisolvon",
            "Grangel",
            "Yumangel",
            "Yumangel",
            "Oresol",
            "Ketovazol",
            "Fugacar",
            "Fugacar",
            "Milian",
            "Biafine",
            "Stugeron",
            "Bilomag",
            "Rowatinex",
            "Tanakan",
            "Ginkgo",
            "Fatig",
            "Ginkor",
            "Diosmin",
            "Hasanflon",
            "Dosaff",
            "Lubirine",
            "Forlax",
            "Ovalax",
            "Magne",
            "Enervon",
            "Surbex",
            "Dorocan",
            "Vitamin",
            "Lacbiosyn",
            "Loperamid",
            "Loperamide",
            "Bioflora",
            "Smecta",
            "Grafort",
            "Smecgim",
            "Lacteol",
            "Imodium",
            "Smecta",
            "Telfor",
            "Antacil",
            "Crila",
            "Enpovid",
            "Vitamin",
            "Multivitamin",
            "Mebizinc",
            "Silygamma",
            "Silymax",
            "Mekotricin",
            "Flexsa",
            "Gellux",
            "Agimfast",
            "Cetirizin",
            "Eftilora",
            "Gimfastnew",
            "Fefasdin",
            "Telfast",
            "Danapha",
            "Loreze",
            "Loratadin",
            "Cetirizin",
            "Zyrtec",
            "Allerfar",
            "Telfast",
            "Anpemux",
            "Picado",
            "Moriamin",
            "Cynaphytol",
            "Panadol",
            "Odistad",
            "Acetab",
            "Hapacol",
            "Coldfed",
            "Dopagan",
            "Efferalgan",
            "Efferalgan",
            "Efferalgan",
            "Hapacol",
            "Hapacol",
            "Hapacol",
            "Hapacol",
            "Mexcold",
            "Paracetamol",
            "Paracetamol",
            "Tydol",
            "Tydol",
            "Tydol",
            "Apibufen",
            "Dibulaxan",
            "Loxfen",
            "Poncityl",
            "Panactol",
            "Tatanol",
            "Tragutan",
            "Biragan",
            "Glotadol",
            "Glotadol",
            "Aspirin",
            "Nurofen",
            "Prospan",
            "Diatabs",
            "Rutin",
            "Arcalion",
            "Eucalyptin",
            "Enterogermina",
            "Enterogermina",
            "Silybean",
            "Kremil",
            "Alaxan",
            "Pruzena",
            "Lorastad",
            "Mitux",
            "Chophytol",
            "Ambroxol",
            "Biolac",
            "Betadine",
            "Vrohto",
            "VRohto",
            "Daigaku",
            "Vrohto",
            "Eyemiru",
            "Systane",
            "Systane",
            "Refresh",
            "Efticol",
            "Tears",
            "VRohto",
            "Vrohto",
            "VRohto",
            "Naphazoline",
            "Otrivin",
            "Rhinex",
            "Devomir",
            "Vitamin",
            "Bestrip",
            "Choliver",
            "Tragutan",
            "Betadine",
            "Medoral",
            "Albendazol",
            "Zentel",
            "Mitux",
            "Postinor",
            "Vitamin",
            "Berberin",
            "Decolgen",
            "Decolgen",
            "Dolfenal",
            "Dentanalgi",
            "Adazol",
            "Varogel",
            "Cystine",
            "Smecta",
            "Clorpheniramin",
            "Eprazinone",
            "Daflon",
            "Suncurmin",
            "Actis",
            "Nabifar",
            "Nasonex",
            "Salonpas",
            "Betadine",
            "Panthenol",
            "Tiger",
            "Tiger",
            "Tiger",
            "Tiger",
            "Tiger",
            "Tocimat",
            "Topbrain",
            "Trineuron",
            "Tydol",
            "Usaallerz",
            "VacoFlon",
            "Vasoclean",
            "Venrutine",
            "Strepsils",
            "Strepsils",
            "Strepsils",
            "Fluomizin",
            "Zolomax",
            "Shinpoong",
            "Haisamin",
            "Strepsils",
            "Strepsils",
            "Strepsils",
            "Strepsils",
            "Strepsils",
            "Strepsils",
            "Prospan",
            "Phytilax",
            "Berocca",
            "Hapacol",
            "CalSource",
            "Vilanta",
            "Vitamin",
            "Vitamin",
            "Vitamin",
            "Vitamin",
            "Vitamin",
            "Xoang",
            "Xylobalan",
            "Xylozin",
            "Xylozin",
            "Zinoprody",
            "Acecyst",
            "AGINTIDIN",
            "Antivomi",
            "CLANOZ",
            "FUBENZON",
            "HAPACOL",
            "Loratadin",
            "Mypara",
            "Natriclorid",
            "PhilUnimeton",
            "TELFOR",
            "Unafen",
            "Acenocoumarol",
            "Acnotin",
            "Acyclovir",
            "Aescin",
            "Agiclovir",
            "AgiEtoxib",
            "Agifuros",
            "Agihistine",
            "Agimdogyl",
            "Agimoti",
            "Agimstan",
            "AgiMycob",
            "Agirovastin",
            "Agirovastin",
            "Agitritine",
            "Aldoric",
            "Alenta",
            "Allergex",
            "Alphachymotrypsine",
            "Alphachymotrypsin",
            "Alpha",
            "Althax",
            "Amedolfen",
            "Aminazin",
            "Amlodac",
            "Amlodipin",
            "Amlodipin",
            "Amlodipine",
            "Amlodipine",
            "Amlodipine",
            "Ampicillin",
            "Anaferon",
            "Apitim",
            "Apival",
            "Aspirin",
            "Astmodil",
            "Aszolzoly",
            "Aticef",
            "Aticef",
            "Atorlip",
            "Atorlip",
            "Atorvastatin",
            "Auclanityl",
            "Auclanityl",
            "Augbactam",
            "Augbidil",
            "Augmex",
            "Augmex",
            "Augtipha",
            "Augtipha",
            "Avamys",
            "Axitan",
            "Azarga",
            "Azicine",
            "Azithromycin",
            "Azithromycin",
            "Zaromax",
            "Azodra",
            "Bambec",
            "Bambuterol",
            "Banitase",
            "Beatil",
            "Belara",
            "Berodual",
            "Beroxib",
            "Betahistine",
            "Betaloc",
            "Betex",
            "Bicebid",
            "Bidacin",
            "Bihasal",
            "Biocemet",
            "Biresort",
            "Biscapro",
            "Bisoplus",
            "Bisoprolol",
            "Bisostad",
            "Bocalex",
            "Bostacet",
            "Cefaclor",
            "Cefaclor",
            "Zitromax",
            "Bozypaine",
            "Bromanase",
            "Camoas",
            "Canvey",
            "Captopril",
            "Cardilopin",
            "Cardioton",
            "Carsantin",
            "Casilas",
            "Cavinton",
            "Cefadroxil",
            "Cefatam",
            "Cefbuten",
            "Cefcenat",
            "Haginir",
            "Cefdinir",
            "Cefdinir",
            "Cefixim",
            "Cefixime",
            "Cefpodoxim",
            "Cefpodoxime",
            "Cefprozil",
            "Cefprozil",
            "Ceftanir",
            "Celecoxib",
            "Celosti",
            "Cepoxitil",
            "CEREBROLYSIN",
            "Chitogast",
            "CHYMODK",
            "Ciclevir",
            "Cimetidin",
            "Cimetidine",
            "Cinnarizine",
            "Clabact",
            "Claminat",
            "Claminat",
            "Claminat",
            "Clarithromycin",
            "Colchicin",
            "Colchicine",
            "Combizar",
            "Concor",
            "Coperil",
            "Coveram",
            "Coversyl",
            "Coxileb",
            "Cozaar",
            "Crestinboston",
            "Crocin",
            "CTToren",
            "Cyclo",
            "Dalekine",
            "Daleston",
            "Daniele",
            "Dantuoxin",
            "Daquetin",
            "Daygra",
            "Dexalevo",
            "Dexclorpheniramin",
            "Dexipharm",
            "Diamicron",
            "Diaprid",
            "Diclofenac",
            "Diclofenac",
            "Ciclevir",
            "Diflucan",
            "Difuzit",
            "Diovan",
            "Disthyrox",
            "Diurefar",
            "Domecor",
            "Dophazolin",
            "Dorobay",
            "Doroclor",
            "Dorocodon",
            "Dorocron",
            "Dorodipin",
            "Dorotor",
            "Dorotril",
            "Dorover",
            "Dozidine",
            "Dozinco",
            "Dudencer",
            "Cravit",
            "Cravit",
            "Oflovid",
            "Alegysal",
            "Travatan",
            "Timolol",
            "Betadine",
            "Aquadetrim",
            "Duoplavin",
            "Duotrav",
            "Ednyt",
            "Efexor",
            "Efodyl",
            "Efodyl",
            "Eliquis",
            "Eltium",
            "Emanera",
            "Emycin",
            "Enalapril",
            "Epegis",
            "Esolona",
            "Esomaxcare",
            "Esoragim",
            "Esoragim",
            "Esseil",
            "Exforge",
            "Expas",
            "Ezvasten",
            "Fabamox",
            "Fatodin",
            "Feburic",
            "Ferlatum",
            "Flixonase",
            "Flodicar",
            "Fluotin",
            "Fosmicin",
            "Fucicort",
            "Fucidin",
            "Furosemide",
            "Gaberon",
            "Galanmer",
            "Galvus",
            "Galvus",
            "Gefbin",
            "Erylik",
            "Eighteen",
            "Daktarin",
            "Gensilron",
            "Gentriboston",
            "Gentri",
            "Glesoz",
            "Glimepiride",
            "GliritDHG",
            "GliritDHG",
            "Glisan",
            "Glocip",
            "Glucofine",
            "Glucofine",
            "Glucophage",
            "Glucophage",
            "Glucophage",
            "Gludipha",
            "Glumeform",
            "Glumeform",
            "Glutoboston",
            "Griseofulvin",
            "Hadugast",
            "Hafenthyl",
            "Hafixim",
            "Hafixim",
            "Haginat",
            "Haginat",
            "Haginat",
            "HAGINIR",
            "Haginir",
            "Hagizin",
            "Haiblok",
            "Haloperidol",
            "Hapenxin",
            "Hasanbest",
            "Hasanclar",
            "Hepbest",
            "Hidrasec",
            "Hiteenall",
            "Hiteen",
            "Maxitrol",
            "Flarex",
            "Hyvalor",
            "Hyvalor",
            "Iboten",
            "Ibutop",
            "Ihybes",
            "Imefed",
            "Imidagi",
            "Infartan",
            "Invel",
            "Irbesartan",
            "Itranstad",
            "Ivermectin",
            "Janumet",
            "Janumet",
            "Jardiance",
            "Jardiance",
            "Jardiance",
            "Jardiance",
            "Kasiod",
            "Kavasdin",
            "Erythromycin",
            "Acyclovir",
            "Erythormycin",
            "Cortibion",
            "Fucidin",
            "Dibetalic",
            "Terfuzol",
            "Acyclovir",
            "Forsancort",
            "Silkeron",
            "Betaderm",
            "Kipel",
            "Kipel",
            "Klamentin",
            "Klamentin",
            "Klamentin",
            "Kuplevotin",
            "Lacipil",
            "Lamictal",
            "LANGITAX",
            "Lazibet",
            "Lecifex",
            "Leolen",
            "Lepigin",
            "Letdion",
            "Levodhg",
            "Levofloxacin",
            "Levoleo",
            "Lilonton",
            "Lincomycin",
            "Lipagim",
            "Lipagim",
            "Lipanthyl",
            "Lipistad",
            "Lipitor",
            "Lisinopril",
            "Livorax",
            "Locgoda",
            "Lomexin",
            "Losartan",
            "Losartan",
            "Lostad",
            "Martaz",
            "Marvelon",
            "Marvelon",
            "Maxdotyl",
            "Maxxprolol",
            "Mebaal",
            "Mebilax",
            "Mecefix",
            "Medexa",
            "Medexa",
            "Medisolone",
            "Meditrol",
            "Medlon",
            "Medrol",
            "Medskin",
            "Medskin",
            "Medskin",
            "Medskin",
            "Meiact",
            "Meiact",
            "Meloxicam",
            "Menison",
            "Menison",
            "Mepoly",
            "Mepraz",
            "Meseca",
            "Methorphan",
            "Methylprednisolon",
            "Methylprednisolon",
            "Metobra",
            "MEYERSILIPTIN",
            "Mezacosid",
            "Micardis",
            "Mirastad",
            "Misoprostol",
            "Mobimed",
            "Mobimed",
            "Morif",
            "Mosad",
            "Motilium",
            "Mydocalm",
            "Myspa",
            "Natrilix",
            "Natrixam",
            "Natrixam",
            "Nazoster",
            "Nebicard",
            "Nebicard",
            "Nebivolol",
            "Nesteloc",
            "Nesulix",
            "Neubatel",
            "Nevanac",
            "Nexium",
            "Nifedipin",
            "Nifedipin",
            "Nifin",
            "Nisten",
            "Normodipine",
            "Novomycine",
            "Nucleo",
            "OFBEXIM",
            "Ofloxacin",
            "Ofmantine",
            "Ofmantine",
            "Ofmantine",
            "Olanxol",
            "Omeprazol",
            "Omeprazole",
            "Omeraz",
            "Onsmix",
            "Daygra",
            "Januvia",
            "Unasyn",
            "Opesinkast",
            "Oracortia",
            "Osarstad",
            "Ospamox",
            "Otipax",
            "Ovumix",
            "Panfor",
            "Pantoloc",
            "Pantoprazol",
            "Pantostad",
            "Pentasa",
            "Peruzi",
            "Pharmox",
            "PhilDomina",
            "Philurso",
            "Piracetam",
            "Piracetam",
            "Piromax",
            "Plendil",
            "Polygynax",
            "Posod",
            "Pradaxa",
            "Pradaxa",
            "Prazopro",
            "Prazopro",
            "Prednison",
            "Prednison",
            "Prednison",
            "Predsantyl",
            "Premilin",
            "Pricefil",
            "Primolut",
            "Procoralan",
            "Progynova",
            "Pycip",
            "PymeAZI",
            "Pymenospain",
            "Pyzacar",
            "Rabepagi",
            "Rabeto",
            "Rabicad",
            "Rabicad",
            "Raxium",
            "Regulon",
            "Repraz",
            "Risperdal",
            "ROBESTATINE",
            "Rocimus",
            "Rocimus",
            "Rodilar",
            "Rosuvas",
            "Rovas",
            "Rovas",
            "Rupafin",
            "Seroquel",
            "Sezstad",
            "Shinclop",
            "Shinpoong",
            "Sifrol",
            "Silkron",
            "Singulair",
            "Singulair",
            "Atussin",
            "Skinrocin",
            "Snapcef",
            "SOSLac",
            "Sovalimus",
            "Sovalimus",
            "Sovepred",
            "Spacmarizine",
            "Spasticon",
            "Spiramycin",
            "Spiromide",
            "Splozarsin",
            "Staclazide",
            "Stadnex",
            "Stadnolol",
            "Stadovas",
            "Stamlo",
            "Sterogyl",
            "Sucrafil",
            "Sulcilat",
            "Sulpirid",
            "Sunapred",
            "Sundronis",
            "Symbicort",
            "Tearbalance",
            "Tefostad",
            "Teginol",
            "Telmisartan",
            "Tenofovir",
            "Terpincold",
            "Tetracyclin",
            "Kaleorid",
            "Tozinax",
            "Neutri",
            "Hidrasec",
            "Hidrasec",
            "SOSCough",
            "Lyrica",
            "Tanatril",
            "Tanatril",
            "Aspilets",
            "Lodimax",
            "Lodimax",
            "Alpha",
            "Medisolone",
            "Prednisolone",
            "Prednison",
            "Singulair",
            "Jasunny",
            "Berodual",
            "Zestoretic",
            "Doxycyclin",
            "Gluzitop",
            "Glucophage",
            "Metformin",
            "Metformin",
            "Xigduo",
            "Glucophage",
            "Glucovance",
            "Enoti",
            "Uruso",
            "Pizar",
            "Pizar",
            "Allopurinol",
            "Symbicort",
            "Crocin",
            "Crocin",
            "Pycalis",
            "Amlor",
            "Captopril",
            "Lostad",
            "Irbesartan",
            "Micardis",
            "Coveram",
            "Tegretol",
            "Diacerein",
            "Optipan",
            "Celebrex",
            "Coveram",
            "Clealine",
            "Stadnex",
            "Nexium",
            "Nexium",
            "Rabestad",
            "Klenzit",
            "Klenzit",
            "Ventolin",
            "Cotrimoxazole",
            "Stadlofen",
            "Levothyrox",
            "Carduran",
            "Transamin",
            "Cebastin",
            "Kingdomin",
            "Trifungi",
            "Methycobal",
            "Doncef",
            "Cefalexin",
            "Dalacin",
            "Augmentin",
            "Augmentin",
            "Clamoxyl",
            "Scanax",
            "OpeCipro",
            "Ciprobay",
            "Dorogyne",
            "Vitamin",
            "Artrodar",
            "Progestogel",
            "Stresam",
            "Dextromethorphan",
            "Levomepromazin",
            "Ultracet",
            "Venlafaxine",
            "Sifrol",
            "Arcoxia",
            "Arcoxia",
            "Arcoxia",
            "Amoxycillin",
            "Bifumax",
            "Tetracyclin",
            "Cephalexin",
            "Cephalexin",
            "Vastarel",
            "Zinnat",
            "Rocaltrol",
            "Amoxicillin",
            "Stadexmin",
            "Mibetel",
            "Cozaar",
            "Transamin",
            "Betaserc",
            "Dexamethasone",
            "Mutecium",
            "Betaserc",
            "Agicetam",
            "Topamax",
            "Dromasm",
            "Ventolin",
            "Combivent",
            "Materazzi",
            "Verospiron",
            "Thyrozol",
            "Thyrozol",
            "Komboglyze",
            "Perglim",
            "Glucofast",
            "Gliclazid",
            "Forxiga",
            "Forxiga",
            "Metformin",
            "Diamicron",
            "Panfor",
            "Trajenta",
            "Tranagliptin",
            "Glucovance",
            "Glucophage",
            "Siofor",
            "Duphaston",
            "Debridat",
            "Antivic",
            "Lyrica",
            "Nitromint",
            "Cedetamin",
            "Essividine",
            "Trileptal",
            "Depakine",
            "Depakine",
            "Keppra",
            "Tegretol",
            "Newbutin",
            "Flunarizine",
            "Seretide",
            "Dochicin",
            "Nasrix",
            "Pulmicort",
            "Seretide",
            "Zensonid",
            "Montiget",
            "Agifovir",
            "Cidetuss",
            "Spasmomen",
            "Parokey",
            "Betahistin",
            "Neurontin",
            "Cozaar",
            "Myonal",
            "Fosamax",
            "Fosamax",
            "Zedcal",
            "Pariet",
            "Mucosta",
            "Pantoloc",
            "Pariet",
            "Losec",
            "Rebastric",
            "Lansoprazole",
            "Mydrin",
            "Nemydexan",
            "Acnotin",
            "Akinol",
            "Sporal",
            "Acyclovir",
            "Acyclovir",
            "Hepatymo",
            "Amoxicillin",
            "Amoxycilin",
            "Augbactam",
            "Erythromycin",
            "Droxicef",
            "Claminat",
            "Zinnat",
            "Fabamox",
            "Fabamox",
            "Lincomycin",
            "Natrofen",
            "Orenko",
            "Vigentin",
            "Tinidazol",
            "Spulit",
            "Fluconazole",
            "Moxacin",
            "Curam",
            "Klacid",
            "Klacid",
            "Ospexin",
            "Azicine",
            "Medskin",
            "Spirastad",
            "Roxithromycin",
            "Tobramycin",
            "Spiranisol",
            "Ceclor",
            "Klacid",
            "Zinnat",
            "Telzid",
            "Avodart",
            "Harnal",
            "Xatral",
            "Franilax",
            "Methionin",
            "Mezaverin",
            "Adagrin",
            "Adagrin",
            "Cialis",
            "Evadam",
            "Viagra",
            "Priligy",
            "Tadalafil",
            "Viagra",
            "Medrol",
            "Utrogestan",
            "AtorHASAN",
            "Fenostad",
            "Halozam",
            "Digoxin",
            "Piracetam",
            "Allopurinol",
            "Kacetam",
            "Gamalate",
            "Uperio",
            "Veinofytol",
            "Zapnex",
            "Risperdal",
            "Sentipec",
            "Rosuvastatin",
            "Atorvastatin",
            "Lipitor",
            "Crestor",
            "Crestor",
            "Rosuvastatin",
            "Crestor",
            "Lipanthyl",
            "Simvastatin",
            "Simvastatin",
            "Lipanthyl",
            "Agidopa",
            "Aprovel",
            "Atasart",
            "Atasart",
            "Bisoprolol",
            "Betaloc",
            "Betaloc",
            "Isosorbid",
            "Isosorbid",
            "Tenocar",
            "Dopegyt",
            "Felodipin",
            "Hyzaar",
            "Lisonorm",
            "Losartan",
            "Lostad",
            "Coversyl",
            "Telmisartan",
            "Pyzacar",
            "Captopril",
            "Coversyl",
            "Coversyl",
            "Diovan",
            "Enalapril",
            "Zestril",
            "Lisinopril",
            "Telzid",
            "Tovecor",
            "Troysar",
            "Zanedip",
            "Coveram",
            "Concor",
            "Adalat",
            "Viacoram",
            "Exforge",
            "Exforge",
            "Exforge",
            "Micardis",
            "Nebilet",
            "Zestril",
            "Zestril",
            "Piascledine",
            "Voltaren",
            "Seretide",
            "Debby",
            "Lifezar",
            "Zoloft",
            "Omeprazol",
            "Mydocalm",
            "MeteoSpasmyl",
            "Melic",
            "Procoralan",
            "Nootropil",
            "Waisan",
            "Vesicare",
            "Amitriptylin",
            "Singulair",
            "Plavix",
            "Lomac",
            "Ventolin",
            "Medisamin",
            "Cordarone",
            "Apisolvat",
            "Betacylic",
            "Entecavir",
            "Clarithromycin",
            "Voltaren",
            "Magrax",
            "Voltaren",
            "Barole",
            "Pepevit",
            "Gefbin",
            "Avamys",
            "Avelox",
            "Cotrimoxazole",
            "Lansoprazol",
            "Alzole",
            "Tavanic",
            "Brilinta",
            "Imdur",
            "Xarelto",
            "Xarelto",
            "Ketosteril",
            "Tinidazol",
            "Nystatin",
            "Rinofil",
            "Difelene",
            "Nidal",
            "Efferalgan",
            "Etoricoxib",
            "Glotadol",
            "Medcaflam",
            "Sympal",
            "Gabahasan",
            "Etoricoxib",
            "Brexin",
            "Glotadol",
            "Seretide",
            "Janumet",
            "Janumet",
            "Lipitor",
            "Januvia",
            "Janumet",
            "Glimepiride",
            "Glimepirid",
            "Glimepind",
            "Nebivoloirkhouma",
            "Lipistad",
            "Domitazol",
            "Genshu",
            "Esomeprazole",
            "Bilaxten",
            "Azithromycin",
            "Cefatam",
            "Cephalexin",
            "Augmentin",
            "Augmentin",
            "Hagimox",
            "Hapenxin",
            "Klamentin",
            "Macromax",
            "Negacef",
            "Ofloxacin",
            "Pentinox",
            "Bactamox",
            "Bactamox",
            "Diclofen",
            "Erythromycin",
            "Piropharm",
            "Mekocetin",
            "Metronidazol",
            "Prednisolon",
            "Cataflam",
            "Vipredni",
            "Hafenthyl",
            "Ameflu",
            "Zoamco",
            "Zoamco",
            "Biracin",
            "Daivobet",
            "PhilClobate",
            "rostarin",
            "Combigan",
            "TobraDex",
            "Vismed",
            "Tobcol",
            "Ciprofloxacin",
            "Polydeson",
            "Pataday",
            "Sanlein",
            "Azopt",
            "Vigadexa",
            "Vigamox",
            "Ospay",
            "Clopistad",
            "Panangin",
            "Glucobay",
            "Aspirin",
            "Clindastad",
            "Amitriptilin",
            "Dofluzol",
            "Dogtapine",
            "Lamictal",
            "Pracetam",
            "Lampar",
            "Angut",
            "Pyzacar",
            "Rostor",
            "Drosperin",
            "Estraceptin",
            "Drosperin",
            "Rigevidon",
            "Rosepire",
            "Rosepire",
            "Diane",
            "Yasmin",
            "Ameflu",
            "Atussin",
            "Oracortia",
            "Elthon",
            "Coldi",
            "Thylmedi",
            "Ticoldex",
            "Tobidex",
            "Tobrex",
            "Tonagas",
            "Topamax",
            "Trajenta",
            "Trimebutin",
            "Trimetazidine",
            "Trimetazidine",
            "Triplixam",
            "Trosicam",
            "Twynsta",
            "Twynsta",
            "Uperio",
            "Uperio",
            "Urostad",
            "Usasolu",
            "Utrogestan",
            "Vaginax",
            "Vaspycar",
            "Vastec",
            "Vastec",
            "Vasulax",
            "Vectrine",
            "Vedanal",
            "Venlafaxine",
            "Verospiron",
            "Viacoram",
            "Viagra",
            "Viagra",
            "Safaria",
            "Vigentin",
            "Vigentin",
            "Visanne",
            "Vitamin",
            "Vocfor",
            "Wonfixime",
            "Xalgetz",
            "Xamiol",
            "Xarelto",
            "Xelostad",
            "Xigduo",
            "Yuraf",
            "Zanimex",
            "Zapnex",
            "ZidocinDHG",
            "Zinmax",
            "Zitromax",
            "Zlatko",
            "Zoamco",
            "Zopistad",
            "THYLMEDI",
            "Acegoi",
            "Acepron",
            "Acritel",
            "Agimoti",
            "Akitykity",
            "Alanboss",
            "Albenca",
            "Albendazol",
            "Alcool",
            "Alcool",
            "Alegin",
            "Aleradin",
            "Allerdep",
            "Allerphast",
            "Aluminium",
            "Ambroxen",
            "Amepox",
            "Amisea",
            "Antimuc",
            "Anyfen",
            "Apifexo",
            "Aquima",
            "Astymin",
            "Aticizal",
            "Atisyrup",
            "Axcel",
            "Befabrol",
            "Betasiphon",
            "Bibonlax",
            "Bicanma",
            "Bilgrel",
            "Bilodin",
            "Biragan",
            "Bisnol",
            "Bitalvic",
            "Bofit",
            "Bufecol",
            "Cabovis",
            "Cadifast",
            "Calcical",
            "Calcichew",
            "Calci",
            "Calci",
            "Calciumboston",
            "Calcium",
            "CalSource",
            "CalSource",
            "Ecosip",
            "Ecosip",
            "Cetirizine",
            "Chalme",
            "Ciprofloxacin",
            "Circuton",
            "Clanzen",
            "Codepect",
            "Coldacmin",
            "Cossinmin",
            "Cynamus",
            "Cynara",
            "Dacodex",
            "Dacolfort",
            "Yellow",
            "Salonpas",
            "Decolgen",
            "Denatri",
            "Derimucin",
            "Deslora",
            "Deslotid",
            "Destacure",
            "Dexcorin",
            "Didicera",
            "Dilodin",
            "Diosfort",
            "Dioxzye",
            "Dismolan",
            "Dodylan",
            "Natri",
            "Duchat",
            "Systane",
            "Povidine",
            "Orafar",
            "Duodart",
            "Efodyl",
            "Eludril",
            "Enterpass",
            "Epiduo",
            "Esonix",
            "Esserose",
            "Eugica",
            "Extracap",
            "Extra",
            "Febustad",
            "Fegra",
            "Fengshi",
            "Ferimond",
            "Ferricure",
            "Fexnad",
            "FexodineFast",
            "Fexofenadin",
            "Fexofenadin",
            "Fitocoron",
            "Floezy",
            "Fogicap",
            "Fucalmax",
            "Gastrylstad",
            "Gebhart",
            "Gelactive",
            "Gelacmeigel",
            "Contractubex",
            "Belskin",
            "Glomoti",
            "Glomoti",
            "Glotadol",
            "Glucosamin",
            "Glyxambi",
            "Goldgro",
            "Goncal",
            "Gourcuff",
            "Grazyme",
            "Guarente",
            "Gumas",
            "Gyllex",
            "Gysudo",
            "Gysudo",
            "Hagifen",
            "Hameron",
            "Hapacol",
            "Hapacol",
            "Hapacol",
            "Hasancob",
            "Healit",
            "Hemafolic",
            "Heparos",
            "Hirudoid",
            "Histalong",
            "Misanlugel",
            "Rhinocort",
            "Mangino",
            "Hylaform",
            "Ibumed",
            "Ibuprofen",
            "Isaias",
            "Jazxylo",
            "Jasunny",
            "Panthenol",
            "Leivis",
            "Nizoral",
            "Kidneyton",
            "Kimraso",
            "KITNO",
            "Ladyformine",
            "Lantasim",
            "Lepatis",
            "Levoseren",
            "Livercom",
            "Livethine",
            "Livosil",
            "Livsin",
            "Loratadine",
            "Macetux",
            "MAECRAN",
            "Masozym",
            "Medibro",
            "MedSkin",
            "Meken",
            "Melopower",
            "Merika",
            "Meseca",
            "Meyermazol",
            "Mezafen",
            "Micezym",
            "Micomedil",
            "Salonpas",
            "Mongor",
            "Muscino",
            "Muspect",
            "Mylenfa",
            "Myrtol",
            "Nadygan",
            "Natri",
            "Natri",
            "Natri",
            "Natri",
            "Neopeptine",
            "Nidal",
            "Nizoral",
            "Novotane",
            "Nozeytin",
            "Ophazidon",
            "Optive",
            "Oresol",
            "Orlistat",
            "Otibone",
            "Paluzine",
            "Pamagel",
            "Panactol",
            "Pancal",
            "Pancres",
            "Paracetamol",
            "Paracetamol",
            "Paratriam",
            "Parazacol",
            "Parazacol",
            "NadyRosa",
            "Philoyvitan",
            "Phong",
            "Pirolam",
            "Cedipect",
            "Povidon",
            "Promethazine",
            "Queenlife",
            "Restasis",
            "Rhinassin",
            "Rhynixsol",
            "Rowachol",
            "Sacendol",
            "Sacendol",
            "Saferon",
            "Savimetoc",
            "SeoulCigenol",
            "Shine",
            "Siloxogene",
            "Silyhepatis",
            "Silymarin",
            "Lorastad",
            "Theralene",
            "Cisteine",
            "Farzicol",
            "Ambroco",
            "Mediphylamin",
            "Colatus",
            "Sorbitol",
            "Sovegal",
            "Stadmazol",
            "Strepsils",
            "Subtyl",
            "Sungin",
            "Systane",
            "Tataca",
            "Telfadin",
            "Terpincodein",
            "Bilobapro",
            "Vinger",
            "Livsin",
            "Betadine",
            "Zinenutri",
            "Fogyma",
            "Aibezym",
            "Magne",
            "Growsel",
            "Aceffex",
            "Philco",
            "Actapulgite",
            "Meyerfast",
            "Mexcold",
            "Kogimin",
            "Pepsane",
            "Bibonlax",
            "Bosgyno",
            "Hexinvon",
            "Folacid",
            "Fexostad",
            "Paxirasol",
            "Gastropulgite",
            "Lumbrotine",
            "Bivantox",
            "Coliomax",
            "SaviFexo",
            "Paracold",
            "Paracold",
            "Colicare",
            "Magnesium",
            "Trivita",
            "Juvever",
            "Carmanus",
            "Albefar",
            "Magne",
            "Mangistad",
            "Meflavon",
            "Codeforte",
            "Scanneuron",
            "Antesik",
            "Eumovate",
            "Vitasmooth",
            "Rotundin",
            "Sucrahasan",
            "Gastro",
            "Valian",
            "Theralene",
            "Regatonic",
            "Paclovir",
            "Mibeviru",
            "Tocimat",
            "Tadimax",
            "Phong",
            "Frentine",
            "Tuzamin",
            "Tradin",
            "Philiver",
            "Memoril",
            "Bacivit",
            "Agiosmin",
            "Taginko",
            "Sorbitol",
            "Sorbitol",
            "Calciumgeral",
            "Vitamin",
            "Zentomyces",
            "Subtyl",
            "Hamett",
            "Canxi",
            "Calcium",
            "Agimoti",
            "Becoridone",
            "Naupastad",
            "Imexofen",
            "Tottim",
            "Vitamin",
            "Trymo",
            "BIBISO",
            "Mezathin",
            "Vorifend",
            "Clorpheniramin",
            "Levoagi",
            "Xonatrix",
            "Tadaritin",
            "Celerzin",
            "Histalong",
            "Telfast",
            "SOSAllergy",
            "Dixirein",
            "Agirenyl",
            "Centasia",
            "Famela",
            "Bamyrol",
            "Hapacol",
            "Panadol",
            "Tatanol",
            "Protamol",
            "Partamol",
            "Terpin",
            "Dogarlic",
            "Viegan",
            "Colocol",
            "Glotadol",
            "Glotadol",
            "Paracetamol",
            "Colocol",
            "Acehasan",
            "Acehasan",
            "Pectol",
            "Xonatrix",
            "Mucosolvan",
            "Urdoc",
            "Mediclovir",
            "Salymet",
            "Cooldrop",
            "Natri",
            "Otrivin",
            "Argistad",
            "Vitarals",
            "Nautamine",
            "Zinobaby",
            "Nimotop",
            "Boncium",
            "Sitar",
            "AginFolix",
            "Livsin",
            "Povidon",
            "Synatura",
            "Ubiheal",
            "Aspartam",
            "Stadleucin",
            "Vacomuc",
            "Loratadine",
            "Dozoltac",
            "Carbomango",
            "Carbomint",
            "Esomez",
            "Rheumapain",
            "Tylenol",
            "Otilin",
            "Momate",
            "Tizanad",
            "Tizanidin",
            "Totcal",
            "Touxirup",
            "Traluvi",
            "Trancumin",
            "Tritenols",
            "Trizomibe",
            "Ulcersep",
            "Ursopa",
            "Usaallerz",
            "Usaallerz",
            "Uscadigesic",
            "Vacomuc",
            "Vasomin",
            "Venokern",
            "LadyBalance",
            "Strepsils",
            "Tyrotab",
            "Tyrotab",
            "Hasanvit",
            "Redoxon",
            "Vitamin",
            "Vitamin",
            "Vitamin",
            "Vitamin",
            "Vitamin",
            "Vitamin",
            "Xoangspray",
            "Zanicicol",
            "Zedcal",
            "Zinbebe",
            "Zysmas",
            "AGIMEPZOL",
            "Agimol",
            "Bebamol",
            "Bebamol",
            "Bixentin",
            "CETIRIZIN",
            "CLORPHENIRAMIN",
            "COLDFED",
            "Dofexo",
            "Dolarac",
            "Dopagan",
            "Dopagan",
            "Dotoux",
            "Eblamin",
            "Effer",
            "ESKAR",
            "FABALOFEN",
            "Ginkokup",
            "GLOTAMUC",
            "HAPACOL",
            "HAPACOL",
            "KIDVita",
            "Kingloba",
            "Loratadine",
            "Macfor",
            "MAGNESI",
            "MEDIFLUDAY",
            "METID",
            "Minigadine",
            "Natri",
            "Pabemin",
            "PACEGAN",
            "PANACTOL",
            "PARACETAMOL",
            "Paracold",
            "PARAHASAN",
            "Richcalusar",
            "Sedanxio",
            "Sorbitol",
            "TATANOL",
            "TERPIN",
            "VOMINA",
            "Abrocto",
            "Acenocoumarol",
            "AcezinDHG",
            "Acyclovir",
            "Aciclovir",
            "Aclasta",
            "Aclon",
            "Acuvail",
            "Acyclovir",
            "Acyclovir",
            "Acyclovir",
            "Acyclovir",
            "Agicarvir",
            "Agicetam",
            "Agiclovir",
            "Agidecotyl",
            "Agifamcin",
            "Agilecox",
            "Agilosart",
            "Agilosart",
            "Agimlisin",
            "Agimstan",
            "Aginolol",
            "Agiremid",
            "Agisimva",
            "Agostini",
            "Aldan",
            "Alenbe",
            "Allopsel",
            "Alpha",
            "Alphachymotrypsin",
            "Alphadeka",
            "Alphagan",
            "Alpha",
            "Alverin",
            "Amaryl",
            "Ameflu",
            "Amespasm",
            "Amfastat",
            "Amgifer",
            "Amilavil",
            "Amloboston",
            "Amlodipine",
            "Amlor",
            "Amoksiklav",
            "Amoxicilin",
            "Amoxicilin",
            "Anapa",
            "Antilox",
            "Apibrex",
            "Apidra",
            "Arbuntec",
            "Aromasin",
            "Atasart",
            "Calactate",
            "Atelec",
            "Atibeza",
            "Atilair",
            "AtiSalbu",
            "Atisalbu",
            "Atizet",
            "Atobe",
            "Atorvastatin",
            "Atussin",
            "Atzeti",
            "Augtipha",
            "Augtipha",
            "Avamys",
            "Axcel",
            "Ayite",
            "Azaduo",
            "Aziphar",
            "Bactamox",
            "Bamifen",
            "Barcavir",
            "Bastinfast",
            "Bepracid",
            "Beprasan",
            "Berlthyrox",
            "Biacefpo",
            "Biacefpo",
            "Bidiclor",
            "Binozyt",
            "Biocemet",
            "Biscapro",
            "Bisoloc",
            "Bivoeso",
            "Bivolcard",
            "Bonviva",
            "Bosagas",
            "Bospicine",
            "Brodicef",
            "Broncho",
            "Brospecta",
            "Bunchen",
            "Bustidin",
            "Butocox",
            "Calcitriol",
            "Calcium",
            "Caldiol",
            "Canasone",
            "Cancetil",
            "Candesartan",
            "Candesartan",
            "Candesartan",
            "Candid",
            "Candid",
            "Canditral",
            "Canditral",
            "Canzeal",
            "Captagim",
            "Carbaro",
            "Carsil",
            "Catolis",
            "Caviar",
            "Cavinton",
            "Cebastin",
            "Cebrex",
            "Cefaclor",
            "Cefadroxil",
            "Cefadroxil",
            "Cefakid",
            "Cefdinir",
            "Cefixim",
            "Cefixim",
            "Cefpivoxil",
            "Ceftopix",
            "Cefubi",
            "Cefuroxim",
            "Celogot",
            "Ceozime",
            "Cephalexine",
            "Cerecozin",
            "Cerepril",
            "Cetampir",
            "Chimitol",
            "Chlorocina",
            "Cholestin",
            "Chymodk",
            "Chymotase",
            "Ciacca",
            "Cilavef",
            "Cilox",
            "Ciprofloxacin",
            "Ciprofloxacin",
            "Citalopram",
            "Clabact",
            "Clarividi",
            "Clarividi",
            "Clazic",
            "Clinecid",
            "Clopalvix",
            "Clopidogrel",
            "Clyodas",
            "Clyodas",
            "CoAprovel",
            "Cododamed",
            "Coirbevel",
            "Colchicine",
            "Coldko",
            "Colthimus",
            "Coltramyl",
            "Combiwave",
            "Coperil",
            "Corbis",
            "Cortisotra",
            "Cotilam",
            "Cotixil",
            "Cotriseptol",
            "Coveram",
            "Coversyl",
            "Crutit",
            "Cttprozil",
            "Curam",
            "Curam",
            "Cyclindox",
            "Cypdicar",
            "Cyplosart",
            "Dagocti",
            "Dalekine",
            "Damipid",
            "Daquetin",
            "Darinol",
            "Darintab",
            "Dasoltac",
            "Decontractyl",
            "Decontractyl",
            "Deferiprone",
            "Degevic",
            "Deginal",
            "Denesity",
            "Denfer",
            "Depakine",
            "Dermabion",
            "Destor",
            "Devodil",
            "Dexamethasone",
            "Dexilant",
            "Dexilant",
            "Dexlacyl",
            "Diaprid",
            "Diaprid",
            "Dicellnase",
            "Diclofen",
            "Dicsep",
            "Dilatrend",
            "Dilatrend",
            "Diltiazem",
            "Dinara",
            "Diopolol",
            "Diprosalic",
            "Distocide",
            "Disys",
            "Docalciole",
            "Dogrel",
            "Domecor",
            "Domepa",
            "Domever",
            "Doniwell",
            "Dopola",
            "Dopolys",
            "Dorobay",
            "Dorocron",
            "Dorokit",
            "Dorolid",
            "Dorosur",
            "Dorover",
            "Duocetz",
            "Dovocin",
            "Doxycyclin",
            "Drexler",
            "Dronagi",
            "Dropstar",
            "DrotaVerine",
            "Ciloxan",
            "Povidine",
            "Duotrav",
            "Dutabit",
            "Dutaon",
            "Duvita",
            "Ebastin",
            "Ebastine",
            "Ebitac",
            "Ecingel",
            "Ednyt",
            "Efavirenz",
            "Efexor",
            "Eftifarene",
            "Egudin",
            "Egudin",
            "Eldosin",
            "Eliquis",
            "Elpertone",
            "Encorate",
            "Entacron",
            "Entero",
            "Epegis",
            "Erilcar",
            "Erilcar",
            "Erybact",
            "Eryne",
            "Esocon",
            "Espacox",
            "Esseil",
            "Etocox",
            "Etova",
            "Euvioxcin",
            "Expas",
            "Eyexacin",
            "Ezensimva",
            "Ezensimva",
            "Ezenstatin",
            "Ezenstatin",
            "Ezetimibe",
            "Fabafixim",
            "Fabafixim",
            "Fabamox",
            "Fabapoxim",
            "Famopsin",
            "Famotidine",
            "Femara",
            "Femoston",
            "Fenaflam",
            "Fendexi",
            "Fenofibrat",
            "Fenosup",
            "Fixnat",
            "Fixnat",
            "Flexidron",
            "Flumetholon",
            "Flumetholon",
            "Fluomizin",
            "Fluopas",
            "Focgo",
            "Fordia",
            "Fordia",
            "Fosmicin",
            "Friburine",
            "Fugentin",
            "Furosemid",
            "Gabantin",
            "Gabena",
            "Galagi",
            "Galagi",
            "GALAPELE",
            "Galvus",
            "GANLOTUS",
            "Gensler",
            "Gensonmax",
            "Gensonmax",
            "Getino",
            "Getzglim",
            "Getzglim",
            "Getzlox",
            "Getzlox",
            "Gifuldin",
            "Gikorcen",
            "Givet",
            "Glimegim",
            "Glizym",
            "Glockner",
            "Glocor",
            "Glofap",
            "Glotal",
            "Glovitor",
            "Glucofine",
            "Gludipha",
            "Glumerif",
            "Glumerif",
            "GOLZYNIR",
            "Gonal",
            "Goutcolcin",
            "Grandaxin",
            "Gymenyl",
            "Halcort",
            "Haloperidol",
            "Hapacol",
            "Hapacol",
            "Hasancob",
            "Hasanbest",
            "Hasanbest",
            "Hasancob",
            "Hasec",
            "Hatlop",
            "Hemfibrat",
            "Herbesser",
            "Hivent",
            "Homan",
            "Hornol",
            "Hueso",
            "Huhajo",
            "Humalog",
            "Humalog",
            "Huntelaar",
            "Hydrocolacyl",
            "Hypravas",
            "Ihybes",
            "Ihybes",
            "Imanok",
            "Imdur",
            "Imeclor",
            "Imeflox",
            "Imidagi",
            "Imidu",
            "Indopril",
            "Insulatard",
            "Itopride",
            "Ivermectin",
            "Iyafin",
            "Jardiance",
            "Jardiance",
            "Jewell",
            "Jikagra",
            "Kagasdine",
            "Kaldyum",
            "Kamydazol",
            "Katies",
            "Kefcin",
            "Asbesone",
            "Temprosone",
            "Beprosone",
            "Keppra",
            "Kernhistine",
            "Ketosan",
            "Klacid",
            "Klamentin",
            "Kononaz",
            "Kozeral",
            "Ladyvagi",
            "Lamone",
            "Lampar",
            "Lamzidivir",
            "Lantus",
            "Lecifex",
            "Ledvir",
            "Lepigin",
            "Leracet",
            "Lertazin",
            "Leukas",
            "Levina",
            "Levistel",
            "Levistel",
            "Levivina",
            "LevoDHG",
            "Levomepromazin",
            "Levothyrox",
            "Levpiram",
            "Lincomycin",
            "Lipiget",
            "Livact",
            "Livorax",
            "Locinvid",
            "Losapin",
            "Losartan",
            "Losartan",
            "Lotemax",
            "Lumigan",
            "Luvox",
            "Lyodura",
            "Malag",
            "Maropol",
            "Maxigra",
            "Maxxacne",
            "Maxxacne",
            "Maxxcardio",
            "Maxxcardio",
            "Maxxcardio",
            "Mebicefpo",
            "Mecabamol",
            "Mecefix",
            "Mecefix",
            "Mecefix",
            "Mecitil",
            "Mediclophencid",
            "Medskin",
            "Mekocefaclor",
            "Mekopen",
            "Mekozitex",
            "Mercifort",
            "Mestinon",
            "Mesulpine",
            "Methopil",
            "Methorfar",
            "Methotrexate",
            "Methyldopa",
            "Methylprednisolone",
            "Metiny",
            "Metiocolin",
            "Metrima",
            "Metronidazol",
            "Metsav",
            "Meyerbastin",
            "Meyercarmol",
            "Meyercolin",
            "Meyercosid",
            "Meyercosid",
            "Meyerflu",
            "Meyeroxofen",
            "Meyerproxen",
            "Meyersolon",
            "Mezagastro",
            "MEZATHIN",
            "Mezavitin",
            "Miaryl",
            "Mibelcam",
            "Mibelcam",
            "Mibeserc",
            "Mibetel",
            "Micbibleucin",
            "Midasol",
            "Milurit",
            "MinirinMELT",
            "Mirgy",
            "Mirzaten",
            "Misenbo",
            "Modom",
            "Molnupiravir",
            "Moloxcin",
            "Molravir",
            "Monitazone",
            "Moritius",
            "Mosane",
            "Motilium",
            "Moxetero",
            "Moxifloxacin",
            "Moxilen",
            "Mycotrova",
            "Mynarac",
            "Nalgidon",
            "Nanfizy",
            "Natondix",
            "Nefolin",
            "Neocin",
            "Nerazzu",
            "Nerazzu",
            "Nergamdicin",
            "Neubatel",
            "Neupencap",
            "Neurobion",
            "Neurogesic",
            "Nevoloxan",
            "NifeHexal",
            "Nirdicin",
            "Nisten",
            "Nivalin",
            "Nivalin",
            "Nizastric",
            "Noklot",
            "Nolvadex",
            "Nolvadex",
            "Novorapid",
            "Nucleo",
            "Nucoxia",
            "Nufotin",
            "NUPIGIN",
            "Nuradre",
            "Nystafar",
            "Nystafar",
            "Nystatab",
            "Oestrogel",
            "Olangim",
            "Oleanzrapitab",
            "Omeprazole",
            "Omeprazole",
            "Onglyza",
            "Decolic",
            "Flexidron",
            "Maxitrol",
            "Singulair",
            "TanaMisolblue",
            "OpeAzitro",
            "Opxil",
            "Orasten",
            "Ospamox",
            "Ospen",
            "Ostagi",
            "Ostebon",
            "Panactol",
            "Panfor",
            "Panfor",
            "Pantagi",
            "Pantonix",
            "Paratramol",
            "Parocontin",
            "Pentasa",
            "Penveril",
            "Perglim",
            "Perindastad",
            "Permixon",
            "Pesancort",
            "Pfertzel",
            "PHAMZOPIC",
            "PHARCAVIR",
            "Pharmaclofen",
            "Pharmox",
            "Phenytoin",
            "Philderma",
            "Philmadol",
            "Picaroxin",
            "Pinclos",
            "Piracetam",
            "Piroxicam",
            "Plavix",
            "Plendil",
            "Pletaal",
            "Pradaxa",
            "Prasogem",
            "Prednisolon",
            "Prednisoion",
            "Pregabalin",
            "Pricefil",
            "Primperan",
            "Progendo",
            "Progentin",
            "Promethazin",
            "Propain",
            "PYCALIS",
            "Pyclin",
            "Pyfaclor",
            "Pymetphage",
            "Pyrazinamide",
            "Qapanto",
            "Qbisalic",
            "Quantopic",
            "Queitoz",
            "Queitoz",
            "Quineril",
            "Ramasav",
            "Ramcamin",
            "Ramipril",
            "Ratidin",
            "Ravenell",
            "Ravenell",
            "Rebamipide",
            "Regofa",
            "Relinide",
            "Remebentin",
            "Remecilox",
            "Remeclar",
            "Remeron",
            "Restoril",
            "Retacnyl",
            "Revolade",
            "Rhumacap",
            "Richstatin",
            "Rifampicin",
            "Ringer",
            "Risdontab",
            "Risperidon",
            "Rivaxored",
            "Rocamux",
            "Rodogyl",
            "Rostor",
            "Roticox",
            "Rotinvast",
            "Rotunda",
            "Roxithromycin",
            "Roxithromycin",
            "Ryzodeg",
            "Sadapron",
            "Salgad",
            "Samaca",
            "Samsca",
            "Telmisartan",
            "Savprocal",
            "Sekaf",
            "Semirad",
            "Semozine",
            "Serenata",
            "Seroquel",
            "Seroquel",
            "Setblood",
            "Sibelium",
            "Sifrol",
            "Silvasten",
            "Silverzinc",
            "Sinrigu",
            "Siofor",
            "Ameflu",
            "Sitar",
            "Sizodon",
            "Smodir",
            "Sosvomit",
            "Spasmaverine",
            "Spasvina",
            "Spiolto",
            "Spiramycin",
            "Spiranisol",
            "Spiromide",
            "Staclazide",
            "Stasamin",
            "Statripsine",
            "STIROS",
            "Stomex",
            "Sukanlov",
            "Sulcilat",
            "Sulfaganin",
            "Sulpiride",
            "Sumiko",
            "Sunmesacol",
            "Sunoxitol",
            "Sunsizopin",
            "Supodatin",
            "Sutagran",
            "SYMBICORT",
            "Synapain",
            "Syndopa",
            "Tacropic",
            "Tacroz",
            "Tacroz",
            "Tadalafil",
            "Tadimax",
            "Taflotan",
            "Talliton",
            "Talliton",
            "Tamiflu",
            "Tanagel",
            "Tazilex",
            "Tedini",
            "Tegrucil",
            "Tegrucil",
            "Telma",
            "Telma",
            "Telma",
            "Telmisartan",
            "Temptcure",
            "Tenamox",
            "Tenfovix",
            "Tenoxicam",
            "Tenoxil",
            "Tetracain",
            "Itamelagin",
            "Thiogamma",
            "Thioridazin",
            "Mycomycen",
            "Neutasol",
            "Betamethason",
            "Beprosone",
            "Protopic",
            "Conipa",
            "Zinnat",
            "Deonas",
            "Adrenoxyl",
            "Plendil",
            "Aceclofenac",
            "Warfarin",
            "Europlin",
            "Ceclor",
            "Methocarbamol",
            "Xylocaine",
            "Flixotide",
            "Deruff",
            "Komboglyze",
            "Mixtard",
            "Glucophage",
            "Trajenta",
            "Usabetic",
            "Glucophage",
            "Bentarcin",
            "Seretide",
            "Nicomen",
            "Dermacol",
            "Flucomedil",
            "Madopar",
            "Bihasal",
            "Losartan",
            "Telmisartan",
            "Betaloc",
            "Betaloc",
            "Vastarel",
            "Posisva",
            "Citalopram",
            "Dermovate",
            "Erythromycin",
            "Ceforipin",
            "Anginovag",
            "Augbactam",
            "Biseptol",
            "Carbamazepine",
            "Gloverin",
            "Rhumenol",
            "Idarac",
            "Clorpromazin",
            "Kidsolon",
            "Tiram",
            "Herbesser",
            "Euroxil",
            "Glotadol",
            "Pasquale",
            "Zoamco",
            "Valsartan",
            "Pracetam",
            "Propranolol",
            "Stadpizide",
            "Neuractine",
            "Olanstad",
            "Lopigim",
            "Drotaverin",
            "Drotusc",
            "Zencombi",
            "Mezamazol",
            "Acarbose",
            "Meyerviliptin",
            "Galvus",
            "Metsav",
            "Glibenclamid",
            "Sparenil",
            "Myomethol",
            "Neuralmin",
            "Basocholine",
            "Bazato",
            "Nadecin",
            "Bluecezine",
            "Atheren",
            "Thenadin",
            "Itametazin",
            "Sakuzyal",
            "Gabapentin",
            "Gabapentin",
            "Neuronstad",
            "Phenytoin",
            "Promag",
            "Valmagol",
            "Trileptal",
            "Spiriva",
            "Neotazin",
            "Trimpol",
            "Reinal",
            "Herbesser",
            "Sarariz",
            "Zalenka",
            "Stafloxin",
            "Tydol",
            "Tatanol",
            "Meyerlukast",
            "Opesinkast",
            "Danapha",
            "Opeverin",
            "Seodeli",
            "Cyclogest",
            "Lacipil",
            "Lantus",
            "Glucobay",
            "Dompenyl",
            "Ethambutol",
            "Mysobenal",
            "Risenate",
            "Rexcal",
            "Evaldez",
            "Barole",
            "Melankit",
            "Gastrolium",
            "Mezapid",
            "Sovasol",
            "Terbisil",
            "Acyclovir",
            "Agiclovir",
            "Aluvia",
            "Agiroxi",
            "Ardineclav",
            "Azithromycin",
            "Bactamox",
            "Bactamox",
            "Bilclamos",
            "Novomycine",
            "Bifucil",
            "Broncocef",
            "Cefaclor",
            "Cefaclor",
            "Mekocefaclor",
            "Bravine",
            "Cefadroxil",
            "Firstlexin",
            "Forlen",
            "Getzlox",
            "Incarxol",
            "Cefimbrano",
            "Mulpax",
            "Klavunamox",
            "Novofungin",
            "Orelox",
            "Novogyl",
            "Rovagi",
            "Rovas",
            "Supertrim",
            "Tobramycin",
            "Travinat",
            "Flagyl",
            "Vigentin",
            "Vigentin",
            "Nystatin",
            "Betnapin",
            "Bloci",
            "Dinpocef",
            "Rapiclav",
            "Vincerol",
            "Minirin",
            "Profertil",
            "Primolut",
            "Ultibro",
            "Thiazifar",
            "Gynoflor",
            "Alphausar",
            "Methionine",
            "Methionine",
            "Decolic",
            "Hasancetam",
            "Maxxsat",
            "Mitalis",
            "Temptcure",
            "Orgametril",
            "Cinasav",
            "Statinagi",
            "Piracetam",
            "Davinfort",
            "Coneulin",
            "Ticonet",
            "Flutonin",
            "Vicetin",
            "Piracefti",
            "Pomatat",
            "Digorich",
            "Autifan",
            "Colestrim",
            "Rishon",
            "Richstatin",
            "Devastin",
            "Surotadina",
            "Aprovel",
            "Colosar",
            "Cozaar",
            "Atenolol",
            "Ambidil",
            "Egilok",
            "Egilok",
            "Hyperium",
            "Irbesartan",
            "Losartan",
            "Losartan",
            "Usasartim",
            "Perindopril",
            "Getvilol",
            "Comegim",
            "Ebitac",
            "Mezathion",
            "Tovecor",
            "Usasartim",
            "Valazyd",
            "Mibelet",
            "Fenoflex",
            "Zafular",
            "Ganfort",
            "MesHanon",
            "Saferon",
            "Golcoxib",
            "Ceozime",
            "Spinolac",
            "Lifezar",
            "Vascam",
            "Paolucci",
            "Piracetam",
            "Ciheptal",
            "Nooapi",
            "Efexor",
            "Stablon",
            "Mecosol",
            "Apidom",
            "Kacetam",
            "Alsiful",
            "Vocfor",
            "Liposic",
            "Zonaxson",
            "Agimoti",
            "Agdicerin",
            "Zeprilnas",
            "Casodex",
            "Arimidex",
            "Eurozitum",
            "Novofungin",
            "Fellaini",
            "Potriolac",
            "Neometin",
            "Toulalan",
            "Esomeprazole",
            "Domreme",
            "Mecefix",
            "Melevo",
            "Meyerdefen",
            "Keflafen",
            "Meloxicam",
            "Stadxicam",
            "Trosicam",
            "Meloxicam",
            "Melankit",
            "Esomaxcare",
            "Wolske",
            "Docefnir",
            "Pyfaclor",
            "Ceclor",
            "Katrypsin",
            "Scolanzo",
            "Benate",
            "Alphachymotrypsin",
            "Levoquin",
            "Fuxicure",
            "Polydexa",
            "Doaspin",
            "Mezapizin",
            "Oliveirim",
            "Smart",
            "Smart",
            "Smart",
            "Usalukast",
            "Opesinkast",
            "Lidocain",
            "Conoges",
            "Aceclonac",
            "Travicol",
            "Kemiwan",
            "Etodagim",
            "Parocontin",
            "Hemol",
            "Busfan",
            "Meyercarmol",
            "Meyerison",
            "Asthmatin",
            "Dextromethorphan",
            "Terpin",
            "Hismedan",
            "Dryches",
            "Migomik",
            "Malag",
            "Amribazin",
            "Augbidil",
            "Augxicine",
            "Bluemoxi",
            "Cefadroxil",
            "Cefadroxil",
            "Ceftopix",
            "Cetraxal",
            "Clarithromycin",
            "Garosi",
            "Hapenxin",
            "Levocide",
            "Midaxin",
            "Neostyl",
            "Nulesavir",
            "Rifampicin",
            "Augbactam",
            "Diclofenac",
            "Glemont",
            "Kaflovo",
            "Novafex",
            "Trimexazol",
            "Pecrandil",
            "Thioserin",
            "Spinolac",
            "Cloramphenicol",
            "Dexamoxi",
            "Neciomex",
            "Daivonex",
            "Flucinar",
            "Oflovid",
            "Gentamicin",
            "Vitamin",
            "Novynette",
            "Flumetholon",
            "Colflox",
            "GENTAMICIN",
            "Neodex",
            "Torexvis",
            "Nikoramyl",
            "Montiget",
            "Montiget",
            "Hemopoly",
            "Aricept",
            "Mephenesin",
            "Myolaxyl",
            "Glimegim",
            "Glimepiride",
            "Trajenta",
            "Dogastrol",
            "Mebsyn",
            "Omesel",
            "Urdoc",
            "Stadovas",
            "Cilzec",
            "Clopias",
            "Egilok",
            "Eslatinb",
            "Medisamin",
            "Tormeg",
            "Tilhasan",
            "Mibedatril",
            "Rosina",
            "Hasaderm",
            "Metronidazole",
            "Onglyza",
            "Carbotrim",
            "Mecefix",
            "Thyperopa",
            "Thytodux",
            "Timmak",
            "Tisercin",
            "Tolucombi",
            "Topdinir",
            "Toraxim",
            "Toropi",
            "Toujeo",
            "Trajenta",
            "Triatec",
            "Tributel",
            "Trihexyphenidyl",
            "Trinitrina",
            "Trinopast",
            "Triplixam",
            "Triplixam",
            "Tritace",
            "Troxevasin",
            "Uptiv",
            "Ursachol",
            "Ursobil",
            "Ursoliv",
            "Uruso",
            "Usaralphar",
            "USASartim",
            "Valiera",
            "Valsacard",
            "Valsacard",
            "Valygyno",
            "Vancomycin",
            "Vashasan",
            "Vaslor",
            "Vasmetine",
            "Vazigoc",
            "Verni",
            "Vexinir",
            "Victoza",
            "Vigorito",
            "Vilget",
            "Vintrypsine",
            "Virclath",
            "Vitamin",
            "Vitol",
            "Vitraclor",
            "Vivace",
            "Vixcar",
            "Vizicin",
            "Wedes",
            "Xalexa",
            "Xonatrix",
            "Yawin",
            "Zealargy",
            "Zebacef",
            "Zensalbu",
            "Zensalbu",
            "Zhekof",
            "Zidotex",
            "Ziptum",
            "Zival",
            "Zlatko",
            "Zlatko",
            "Zoloman",
            "Zonafil",
            "Zynadex",
            "Zytovyrin",
            "Goldmycin",
            "Glyceryl",
        ]
        """
            Args:
                df

            Returns:
                character and word features

        """
        special_chars = [
            "&",
            "@",
            "#",
            "(",
            ")",
            "-",
            "+",
            "=",
            "*",
            "%",
            ".",
            ",",
            "\\",
            "/",
            "|",
            ":",
        ]

        # character wise
        (
            n_upper,
            n_spaces,
            n_alpha,
            n_numeric,
            n_special,
            n_quantity_related,
            n_is_drug,
            similarity_scores,
            n_width,
            n_height,
            n_aspect_ratio,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for words in data:
            #
            quantity_related = (
                1
                if (
                    re.search(
                        r"\b\d+\s*viên\b|\bviên\s*\d+\b|\b\d+\s*-\s*viên\b|\bviên\s*-\s*\d+\b",
                        words,
                        flags=re.IGNORECASE,
                    )
                    and len(words.split()) == 2
                )
                else 0
            )
            n_quantity_related.append(quantity_related)

            # Kiểm tra xem từ có phải là tên thuốc hay không
            is_drug = (
                1 if any(drug.lower() in words.lower() for drug in drug_names) else 0
            )
            n_is_drug.append(is_drug)

            # Vectorize text using TfidfVectorizer
            vectorized_name = self.vectorizer.fit_transform([words] + drug_names[1:])
            similarity_score = (vectorized_name[0] * vectorized_name[1:].T).max()
            similarity_scores.append(similarity_score)

            upper, alpha, spaces, numeric, special = 0, 0, 0, 0, 0
            width = len(words)
            height = len(words.split())
            aspect_ratio = width / max(1, height)  # Tránh chia cho 0
            for char in words:
                # for upper letters
                if char.isupper():
                    upper += 1
                # for white spaces
                if char.isspace():
                    spaces += 1
                # for alphabetic chars
                if char.isalpha():
                    alpha += 1
                # for numeric chars
                if char.isnumeric():
                    numeric += 1
                if char in special_chars:
                    special += 1

            n_width.append(width)
            n_height.append(height)
            n_aspect_ratio.append(aspect_ratio)
            n_upper.append(upper)
            n_spaces.append(spaces)
            n_alpha.append(alpha)
            n_numeric.append(numeric)
            n_special.append(special)
            # features.append([n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_digits])

        (
            df["n_upper"],
            df["n_alpha"],
            df["n_spaces"],
            df["n_numeric"],
            df["n_special"],
            df["n_width"],
            df["n_height"],
            df["n_aspect_ratio"],
            df["n_quantity_related"],
            df["n_is_drug"],
            df["similarity_scores"],
        ) = (
            n_upper,
            n_alpha,
            n_spaces,
            n_numeric,
            n_special,
            n_width,
            n_height,
            n_aspect_ratio,
            n_quantity_related,
            n_is_drug,
            similarity_scores,
        )

    def relative_distance(self, export_document_graph=False):
        """
        1) Calculates relative distances for each node in left, right, top  and bottom directions if they exist.
        rd_l, rd_r = relative distances left , relative distances right. The distances are divided by image width
        rd_t, rd_b = relative distances top , relative distances bottom. The distances are divided by image length

        2) Exports the complete document graph for visualization

        Args:
            result dataframe from graph_formation()

        returns:
            dataframe with features and exports document graph if prompted
        """

        df, img = self.df, self.image
        image_height, image_width = self.image.shape[0], self.image.shape[1]
        plot_df = df.copy()
        # Thêm các đặc trưng vị trí và từ lân cận
        n_nearby_words = []

        for index in df["index"].to_list():
            right_index = df.loc[df["index"] == index, "right"].values[0]
            left_index = df.loc[df["index"] == index, "left"].values[0]
            bottom_index = df.loc[df["index"] == index, "bottom"].values[0]
            top_index = df.loc[df["index"] == index, "top"].values[0]

            # check if it is NaN value
            if np.isnan(right_index) == False:
                right_word_left = df.loc[df["index"] == right_index, "xmin"].values[0]
                source_word_right = df.loc[df["index"] == index, "xmax"].values[0]
                df.loc[df["index"] == index, "rd_r"] = (
                    right_word_left - source_word_right
                ) / image_width

                """
                for plotting purposes
                getting the mid point of the values to draw the lines for the graph
                mid points of source and destination for the bounding boxes
                """
                right_word_x_max = df.loc[df["index"] == right_index, "xmax"].values[0]
                right_word_y_max = df.loc[df["index"] == right_index, "ymax"].values[0]
                right_word_y_min = df.loc[df["index"] == right_index, "ymin"].values[0]

                df.loc[df["index"] == index, "destination_x_hori"] = (
                    right_word_x_max + right_word_left
                ) / 2
                df.loc[df["index"] == index, "destination_y_hori"] = (
                    right_word_y_max + right_word_y_min
                ) / 2

            if np.isnan(left_index) == False:
                left_word_right = df.loc[df["index"] == left_index, "xmax"].values[0]
                source_word_left = df.loc[df["index"] == index, "xmin"].values[0]
                df.loc[df["index"] == index, "rd_l"] = (
                    left_word_right - source_word_left
                ) / image_width

            if np.isnan(bottom_index) == False:
                bottom_word_top = df.loc[df["index"] == bottom_index, "ymin"].values[0]
                source_word_bottom = df.loc[df["index"] == index, "ymax"].values[0]
                df.loc[df["index"] == index, "rd_b"] = (
                    bottom_word_top - source_word_bottom
                ) / image_height

                """for plotting purposes"""
                bottom_word_top_max = df.loc[
                    df["index"] == bottom_index, "ymax"
                ].values[0]
                bottom_word_x_max = df.loc[df["index"] == bottom_index, "xmax"].values[
                    0
                ]
                bottom_word_x_min = df.loc[df["index"] == bottom_index, "xmin"].values[
                    0
                ]
                df.loc[df["index"] == index, "destination_y_vert"] = (
                    bottom_word_top_max + bottom_word_top
                ) / 2
                df.loc[df["index"] == index, "destination_x_vert"] = (
                    bottom_word_x_max + bottom_word_x_min
                ) / 2

            if np.isnan(top_index) == False:
                top_word_bottom = df.loc[df["index"] == top_index, "ymax"].values[0]
                source_word_top = df.loc[df["index"] == index, "ymin"].values[0]
                df.loc[df["index"] == index, "rd_t"] = (
                    top_word_bottom - source_word_top
                ) / image_height

            # Thêm đặc trưng vị trí của từ
            df.loc[df["index"] == index, "position"] = index / len(df)

        # replace all tne NaN values with '0' meaning there is nothing in that direction
        df[["rd_r", "rd_b", "rd_l", "rd_t"]] = df[
            ["rd_r", "rd_b", "rd_l", "rd_t"]
        ].fillna(0)

        if export_document_graph:
            for idx, row in df.iterrows():
                # bounding box
                cv2.rectangle(
                    img,
                    (row["xmin"], row["ymin"]),
                    (row["xmax"], row["ymax"]),
                    (0, 0, 255),
                    2,
                )

                if np.isnan(row["destination_x_vert"]) == False:
                    source_x = (row["xmax"] + row["xmin"]) / 2
                    source_y = (row["ymax"] + row["ymin"]) / 2

                    cv2.line(
                        img,
                        (int(source_x), int(source_y)),
                        (
                            int(row["destination_x_vert"]),
                            int(row["destination_y_vert"]),
                        ),
                        (0, 255, 0),
                        2,
                    )

                    text = "{:.3f}".format(row["rd_b"])
                    text_coordinates = (
                        int((row["destination_x_vert"] + source_x) / 2),
                        int((row["destination_y_vert"] + source_y) / 2),
                    )
                    cv2.putText(
                        img,
                        text,
                        text_coordinates,
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        (255, 0, 0),
                        1,
                    )

                    # text_coordinates = ((row['destination_x_vert'] + source_x)/2 , (row['destination_y_vert'] +source_y)/2)

                if np.isnan(row["destination_x_hori"]) == False:
                    source_x = (row["xmax"] + row["xmin"]) / 2
                    source_y = (row["ymax"] + row["ymin"]) / 2

                    cv2.line(
                        img,
                        (int(source_x), int(source_y)),
                        (
                            int(row["destination_x_hori"]),
                            int(row["destination_y_hori"]),
                        ),
                        (0, 255, 0),
                        2,
                    )

                    text = "{:.3f}".format(row["rd_r"])
                    text_coordinates = (
                        int((row["destination_x_hori"] + source_x) / 2),
                        int((row["destination_y_hori"] + source_y) / 2),
                    )
                    cv2.putText(
                        img,
                        text,
                        text_coordinates,
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        (255, 0, 0),
                        1,
                    )

                # cv2.imshow("image", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if not os.path.exists(
                    "D:/Luan Van/Project/med-scan-backend/GCN/figures/graphs"
                ):
                    os.makedirs(
                        "D:/Luan Van/Project/med-scan-backend/GCN/figures/graphs"
                    )

                plot_path = (
                    "D:/Luan Van/Project/med-scan-backend/GCN/figures/graphs/"
                    + self.filename
                    + "docu_graph"
                    ".jpg"
                )
                cv2.imwrite(plot_path, img)

        # drop the unnecessary columns
        df.drop(
            [
                "destination_x_hori",
                "destination_y_hori",
                "destination_y_vert",
                "destination_x_vert",
            ],
            axis=1,
            inplace=True,
        )
        # print(df)
        self.get_text_features(df)

        # # Thêm đặc trưng từ lân cận
        # for index in df["index"].to_list():
        #     left_index = df.loc[df["index"] == index, "left"].values[0]
        #     right_index = df.loc[df["index"] == index, "right"].values[0]

        #     # Kiểm tra nếu có giá trị là 1 trong cột 'n_quantity_related'
        #     if df.loc[df["index"] == index, "n_quantity_related"].values[0] == 1:
        #         # Lấy ngay giá trị right nếu nó trùng với left
        #         if left_index == right_index:
        #             nearby_words = [df.loc[df["index"] == index, "Object"].values[0]]
        #         else:
        #             # Tìm khoảng cách right gần với left của trường đang có giá trị 1 nhất
        #             min_distance = float("inf")
        #             nearest_right_index = None
        #             second_nearest_right_index = None

        #             for other_index in df["index"].to_list():
        #                 if other_index != index:  # Loại bỏ trường đang xét
        #                     other_right_index = df.loc[
        #                         df["index"] == other_index, "right"
        #                     ].values[0]
        #                     if not np.isnan(other_right_index):
        #                         distance = abs(left_index - other_right_index)

        #                         # Nếu trùng với left_index, lấy ngay giá trị right đó
        #                         if distance == 0:
        #                             nearest_right_index = other_index
        #                             break

        #                         if distance < min_distance:
        #                             min_distance = distance
        #                             nearest_right_index = other_index
        #                         elif (
        #                             distance == min_distance
        #                             and other_right_index
        #                             > df.loc[
        #                                 df["index"] == nearest_right_index, "right"
        #                             ].values[0]
        #                         ):
        #                             nearest_right_index = other_index

        #             # Lấy giá trị right lớn hơn trong trường hợp có 2 giá trị right gần nhất
        #             if nearest_right_index is not None:
        #                 nearby_words = [
        #                     df.loc[df["index"] == nearest_right_index, "Object"].values[
        #                         0
        #                     ]
        #                 ]
        #             else:
        #                 nearby_words = []

        #         n_nearby_words.append(nearby_words)
        #     else:
        #         n_nearby_words.append([])  # Không tính toán nearby_words
        # Thêm đặc trưng từ lân cận
        for index in df["index"].to_list():
            left_index = df.loc[df["index"] == index, "left"].values[0]
            right_index = df.loc[df["index"] == index, "right"].values[0]

            # Kiểm tra nếu có giá trị là 1 trong cột 'n_quantity_related'
            if df.loc[df["index"] == index, "n_quantity_related"].values[0] == 1:
                # Lấy ngay giá trị right nếu nó trùng với left
                if left_index == right_index:
                    nearby_words = df.loc[df["index"] == index, "Object"].values[0]
                else:
                    # Tìm khoảng cách right gần với left của trường đang có giá trị 1 nhất
                    min_distance = float("inf")
                    nearest_right_index = None
                    second_nearest_right_index = None

                    for other_index in df["index"].to_list():
                        if other_index != index:  # Loại bỏ trường đang xét
                            other_right_index = df.loc[
                                df["index"] == other_index, "right"
                            ].values[0]
                            if not np.isnan(other_right_index):
                                distance = abs(left_index - other_right_index)

                                # Nếu trùng với left_index, lấy ngay giá trị right đó
                                if distance == 0:
                                    nearest_right_index = other_index
                                    break

                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_right_index = other_index
                                elif (
                                    distance == min_distance
                                    and other_right_index
                                    > df.loc[
                                        df["index"] == nearest_right_index, "right"
                                    ].values[0]
                                ):
                                    nearest_right_index = other_index

                    # Lấy giá trị right lớn hơn trong trường hợp có 2 giá trị right gần nhất
                    if nearest_right_index is not None:
                        nearby_words = df.loc[
                            df["index"] == nearest_right_index, "Object"
                        ].values[0]
                    else:
                        nearby_words = "none"

                n_nearby_words.append(nearby_words)
            else:
                n_nearby_words.append("none")  # Không tính toán nearby_words

        df["n_nearby_words"] = n_nearby_words
        # print(df)
        return df

    # # features calculation
    # def get_text_features(self, df):
    #     """
    #     gets text features

    #     Args: df
    #     Returns: n_lower, n_upper, n_spaces, n_alpha, n_numeric,n_special
    #     """
    #     data = df["Object"].tolist()

    #     """
    #         Args:
    #             df

    #         Returns:
    #             character and word features

    #     """
    #     special_chars = [
    #         "&",
    #         "@",
    #         "#",
    #         "(",
    #         ")",
    #         "-",
    #         "+",
    #         "=",
    #         "*",
    #         "%",
    #         ".",
    #         ",",
    #         "\\",
    #         "/",
    #         "|",
    #         ":",
    #     ]

    #     # character wise
    #     n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_special = (
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #     )

    #     for words in data:
    #         upper, alpha, spaces, numeric, special = 0, 0, 0, 0, 0
    #         for char in words:
    #             # for upper letters
    #             if char.isupper():
    #                 upper += 1
    #             # for white spaces
    #             if char.isspace():
    #                 spaces += 1
    #             # for alphabetic chars
    #             if char.isalpha():
    #                 alpha += 1
    #             # for numeric chars
    #             if char.isnumeric():
    #                 numeric += 1
    #             if char in special_chars:
    #                 special += 1

    #         # n_lower.append(lower)
    #         n_upper.append(upper)
    #         n_spaces.append(spaces)
    #         n_alpha.append(alpha)
    #         n_numeric.append(numeric)
    #         n_special.append(special)
    #         # features.append([n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_digits])

    #     (
    #         df["n_upper"],
    #         df["n_alpha"],
    #         df["n_spaces"],
    #         df["n_numeric"],
    #         df["n_special"],
    #     ) = (n_upper, n_alpha, n_spaces, n_numeric, n_special)

    # def relative_distance(self, export_document_graph=False):
    #     """
    #     1) Calculates relative distances for each node in left, right, top  and bottom directions if they exist.
    #     rd_l, rd_r = relative distances left , relative distances right. The distances are divided by image width
    #     rd_t, rd_b = relative distances top , relative distances bottom. The distances are divided by image length

    #     2) Exports the complete document graph for visualization

    #     Args:
    #         result dataframe from graph_formation()

    #     returns:
    #         dataframe with features and exports document graph if prompted
    #     """

    #     df, img = self.df, self.image
    #     image_height, image_width = self.image.shape[0], self.image.shape[1]
    #     plot_df = df.copy()

    #     for index in df["index"].to_list():
    #         right_index = df.loc[df["index"] == index, "right"].values[0]
    #         left_index = df.loc[df["index"] == index, "left"].values[0]
    #         bottom_index = df.loc[df["index"] == index, "bottom"].values[0]
    #         top_index = df.loc[df["index"] == index, "top"].values[0]

    #         # check if it is NaN value
    #         if np.isnan(right_index) == False:
    #             right_word_left = df.loc[df["index"] == right_index, "xmin"].values[0]
    #             source_word_right = df.loc[df["index"] == index, "xmax"].values[0]
    #             df.loc[df["index"] == index, "rd_r"] = (
    #                 right_word_left - source_word_right
    #             ) / image_width

    #             """
    #             for plotting purposes
    #             getting the mid point of the values to draw the lines for the graph
    #             mid points of source and destination for the bounding boxes
    #             """
    #             right_word_x_max = df.loc[df["index"] == right_index, "xmax"].values[0]
    #             right_word_y_max = df.loc[df["index"] == right_index, "ymax"].values[0]
    #             right_word_y_min = df.loc[df["index"] == right_index, "ymin"].values[0]

    #             df.loc[df["index"] == index, "destination_x_hori"] = (
    #                 right_word_x_max + right_word_left
    #             ) / 2
    #             df.loc[df["index"] == index, "destination_y_hori"] = (
    #                 right_word_y_max + right_word_y_min
    #             ) / 2

    #         if np.isnan(left_index) == False:
    #             left_word_right = df.loc[df["index"] == left_index, "xmax"].values[0]
    #             source_word_left = df.loc[df["index"] == index, "xmin"].values[0]
    #             df.loc[df["index"] == index, "rd_l"] = (
    #                 left_word_right - source_word_left
    #             ) / image_width

    #         if np.isnan(bottom_index) == False:
    #             bottom_word_top = df.loc[df["index"] == bottom_index, "ymin"].values[0]
    #             source_word_bottom = df.loc[df["index"] == index, "ymax"].values[0]
    #             df.loc[df["index"] == index, "rd_b"] = (
    #                 bottom_word_top - source_word_bottom
    #             ) / image_height

    #             """for plotting purposes"""
    #             bottom_word_top_max = df.loc[
    #                 df["index"] == bottom_index, "ymax"
    #             ].values[0]
    #             bottom_word_x_max = df.loc[df["index"] == bottom_index, "xmax"].values[
    #                 0
    #             ]
    #             bottom_word_x_min = df.loc[df["index"] == bottom_index, "xmin"].values[
    #                 0
    #             ]
    #             df.loc[df["index"] == index, "destination_y_vert"] = (
    #                 bottom_word_top_max + bottom_word_top
    #             ) / 2
    #             df.loc[df["index"] == index, "destination_x_vert"] = (
    #                 bottom_word_x_max + bottom_word_x_min
    #             ) / 2

    #         if np.isnan(top_index) == False:
    #             top_word_bottom = df.loc[df["index"] == top_index, "ymax"].values[0]
    #             source_word_top = df.loc[df["index"] == index, "ymin"].values[0]
    #             df.loc[df["index"] == index, "rd_t"] = (
    #                 top_word_bottom - source_word_top
    #             ) / image_height

    #     # replace all tne NaN values with '0' meaning there is nothing in that direction
    #     df[["rd_r", "rd_b", "rd_l", "rd_t"]] = df[
    #         ["rd_r", "rd_b", "rd_l", "rd_t"]
    #     ].fillna(0)

    #     if export_document_graph:
    #         for idx, row in df.iterrows():
    #             # bounding box
    #             cv2.rectangle(
    #                 img,
    #                 (row["xmin"], row["ymin"]),
    #                 (row["xmax"], row["ymax"]),
    #                 (0, 0, 255),
    #                 2,
    #             )

    #             if np.isnan(row["destination_x_vert"]) == False:
    #                 source_x = (row["xmax"] + row["xmin"]) / 2
    #                 source_y = (row["ymax"] + row["ymin"]) / 2

    #                 cv2.line(
    #                     img,
    #                     (int(source_x), int(source_y)),
    #                     (
    #                         int(row["destination_x_vert"]),
    #                         int(row["destination_y_vert"]),
    #                     ),
    #                     (0, 255, 0),
    #                     2,
    #                 )

    #                 text = "{:.3f}".format(row["rd_b"])
    #                 text_coordinates = (
    #                     int((row["destination_x_vert"] + source_x) / 2),
    #                     int((row["destination_y_vert"] + source_y) / 2),
    #                 )
    #                 cv2.putText(
    #                     img,
    #                     text,
    #                     text_coordinates,
    #                     cv2.FONT_HERSHEY_DUPLEX,
    #                     0.4,
    #                     (255, 0, 0),
    #                     1,
    #                 )

    #                 # text_coordinates = ((row['destination_x_vert'] + source_x)/2 , (row['destination_y_vert'] +source_y)/2)

    #             if np.isnan(row["destination_x_hori"]) == False:
    #                 source_x = (row["xmax"] + row["xmin"]) / 2
    #                 source_y = (row["ymax"] + row["ymin"]) / 2

    #                 cv2.line(
    #                     img,
    #                     (int(source_x), int(source_y)),
    #                     (
    #                         int(row["destination_x_hori"]),
    #                         int(row["destination_y_hori"]),
    #                     ),
    #                     (0, 255, 0),
    #                     2,
    #                 )

    #                 text = "{:.3f}".format(row["rd_r"])
    #                 text_coordinates = (
    #                     int((row["destination_x_hori"] + source_x) / 2),
    #                     int((row["destination_y_hori"] + source_y) / 2),
    #                 )
    #                 cv2.putText(
    #                     img,
    #                     text,
    #                     text_coordinates,
    #                     cv2.FONT_HERSHEY_DUPLEX,
    #                     0.4,
    #                     (255, 0, 0),
    #                     1,
    #                 )

    #             # cv2.imshow("image", img)
    #             # cv2.waitKey(0)
    #             # cv2.destroyAllWindows()
    #             if not os.path.exists(
    #                 "D:/Luan Van/Project/med-scan-backend/GCN/figures/graphs"
    #             ):
    #                 os.makedirs(
    #                     "D:/Luan Van/Project/med-scan-backend/GCN/figures/graphs"
    #                 )

    #             plot_path = (
    #                 "D:/Luan Van/Project/med-scan-backend/GCN/figures/graphs/"
    #                 + self.filename
    #                 + "docu_graph"
    #                 ".jpg"
    #             )
    #             cv2.imwrite(plot_path, img)

    #     # drop the unnecessary columns
    #     df.drop(
    #         [
    #             "destination_x_hori",
    #             "destination_y_hori",
    #             "destination_y_vert",
    #             "destination_x_vert",
    #         ],
    #         axis=1,
    #         inplace=True,
    #     )
    #     self.get_text_features(df)
    #     # print(df)
    #     return df


if __name__ == "__main__":
    file = "1"
    connect = Grapher(file)
    # print(type(file))
    G, result, df = connect.graph_formation(export_graph=True)
    # print("result", result)
    df = connect.relative_distance(export_document_graph=True)
    # print(df)
