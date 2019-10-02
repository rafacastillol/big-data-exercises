package nearsoft.academy.bigdata.recommendation;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

public class MovieRecommender {
  private String path;
  private long totalReviews;
  private Map<String, Long> users;
  private Map<Long, String> products;
  private UserBasedRecommender recommender;
  

  public MovieRecommender(String path) throws IOException, TasteException {
    this.path = path;
    this.totalReviews = 0;
    this.users = new HashMap<String, Long>();

    DataModel model = this.loadReviews(path);
    UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
    UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
    this.recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
  }

  private DataModel loadReviews(String path) throws IOException {
    BufferedReader reader;
    InputStream stream = new GZIPInputStream(new FileInputStream(path));
    reader = new BufferedReader(new InputStreamReader(stream, "US-ASCII"));
    String line = reader.readLine();
    BiMap<String, Long> products = HashBiMap.create();

    long nextUserID = 0;
    long nextProductID = 0;

    int state = 0;

    long userID = 0;
    long productID = 0;
    float score = 0.0f;
    FastByIDMap<Collection<Preference>> data = new FastByIDMap<Collection<Preference>>();
    while (line != null) {
      if (line.startsWith("product/productId:")) {
        String productString = line.split(": ")[1];
        state |= 1;

        if (!products.containsKey(productString)) {
          products.put(productString, nextProductID++);
        }
        productID = products.get(productString);
      }
      else if (line.startsWith("review/userId")) {
        String userString = line.split(": ")[1];
        state |= 2;

        if (!this.users.containsKey(userString)) {
          this.users.put(userString, nextUserID++);
        }
        userID = this.users.get(userString);
      }
      else if (line.startsWith("review/score")) {
        score = Float.parseFloat(line.split(": ")[1]);
        state |= 4;
      }

      if (state == 7) {
        this.addPreference(data, userID, productID, score);
        this.totalReviews++;
        if (this.totalReviews % 1000 == 0) {
          System.out.println(this.totalReviews);
        }
        state = 0;
      }
      line = reader.readLine();
    }

    this.products = products.inverse();

    reader.close();

    return new GenericDataModel(GenericDataModel.toDataMap(data, true));
  }

  public void addPreference(
      FastByIDMap<Collection<Preference>> data,
      long userID, long productID, float score) {
    Collection<Preference> prefs = data.get(userID);
    if (prefs == null) {
      prefs = new ArrayList<Preference>(2);
      data.put(userID, prefs);
    }
    Preference pref = new GenericPreference(
      userID,
      productID,
      score
    );
    prefs.add(pref);
  }

  public long getTotalReviews() {
    return this.totalReviews;
  }

  public long getTotalProducts() {
    return this.products.size();
  }

  public long getTotalUsers() {
    return this.users.size();
  }

  public List<String> getRecommendationsForUser(String userID) throws TasteException {
    System.out.println("Getting recs");
    long numericUserID = this.users.get(userID);
    List<String> recs = new ArrayList<String>();
    
    List<RecommendedItem> recommendations = this.recommender.recommend(numericUserID, 3);
    System.out.println("Got recs");
    System.out.println(recommendations);

    List<String> recommendationIDs = new ArrayList<String>(recommendations.size());
    for (RecommendedItem recommendation: recommendations) {
      recommendationIDs.add(this.products.get(recommendation.getItemID()));
    }

    return recommendationIDs;
  }
}